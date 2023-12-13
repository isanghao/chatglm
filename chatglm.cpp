#include "chatglm.h"
#include <algorithm>
#include <codecvt>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <locale>
#include <numeric>
#include <random>
#include <regex>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <unordered_set>

#ifdef __has_include
#if __has_include(<unistd.h>)
#include <unistd.h>
#if defined(_POSIX_MAPPED_FILES)
#include <sys/mman.h>
#endif
#if defined(_POSIX_MEMLOCK_RANGE)
#include <sys/resource.h>
#endif
#endif
#endif

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <io.h>
#include <stdio.h>
#include <windows.h>
#endif

namespace chatglm {

    // ===== ChatGLM-6B =====

    ChatGLMTokenizer::ChatGLMTokenizer(std::string_view serialized_model_proto) {
        const auto status = sp.LoadFromSerializedProto(serialized_model_proto);
        CHATGLM_CHECK(status.ok()) << status.ToString();

        bos_token_id = sp.PieceToId("<sop>");
        eos_token_id = sp.PieceToId("<eop>");
        mask_token_id = sp.PieceToId("[MASK]");
        gmask_token_id = sp.PieceToId("[gMASK]");
        pad_token_id = sp.PieceToId("<pad>");
    }

    std::vector<int> ChatGLMTokenizer::encode(const std::string& text, int max_length) const {
        std::string input = preprocess(text);
        std::vector<int> ids;
        sp.Encode(input, &ids);
        ids.insert(ids.end(), { gmask_token_id, bos_token_id });
        if ((int)ids.size() > max_length) {
            // sliding window: always take the last max_length tokens
            ids.erase(ids.begin(), ids.end() - max_length);
        }
        return ids;
    }

    std::vector<int> ChatGLMTokenizer::encode_history(const std::vector<std::string>& history, int max_length) const {
        std::string prompt = build_prompt(history);
        std::vector<int> input_ids = encode(prompt, max_length);
        return input_ids;
    }

    std::string ChatGLMTokenizer::build_prompt(const std::vector<std::string>& history) {
        CHATGLM_CHECK(history.size() % 2 == 1) << "invalid history size " << history.size();

        std::ostringstream oss_prompt;
        if (history.size() == 1) {
            oss_prompt << history.front();
        }
        else {
            for (size_t i = 0; i < history.size(); i += 2) {
                oss_prompt << "[Round " << i / 2 << "]\n问：" << history[i] << "\n答：";
                if (i < history.size() - 1) {
                    oss_prompt << history[i + 1] << "\n";
                }
            }
        }
        return oss_prompt.str();
    }

    std::string ChatGLMTokenizer::decode(const std::vector<int>& ids) const {
        std::string text;
        sp.Decode(ids, &text);
        text = postprocess(text);
        return text;
    }

    static std::string regex_replace(const std::string& input, const std::regex& regex,
        std::function<std::string(const std::smatch&)> format) {
        std::ostringstream oss;
        int last_index = 0;
        for (auto it = std::sregex_iterator(input.begin(), input.end(), regex); it != std::sregex_iterator(); it++) {
            oss << it->prefix() << format(*it);
            last_index = it->position() + it->length();
        }
        oss << input.substr(last_index);
        return oss.str();
    }

    std::string ChatGLMTokenizer::preprocess(const std::string& text) {
        std::string output;

        // newline token
        {
            static const std::regex newline_regex("\n");
            output = std::regex_replace(text, newline_regex, "<n>");
        }
        // tab token
        {
            static const std::regex tab_regex("\t");
            output = std::regex_replace(output, tab_regex, "<|tab|>");
        }
        // blank tokens
        {
            static const std::regex pattern(R"([ ]{2,80})");
            output = regex_replace(output, pattern, [](const std::smatch& sm) {
                std::ostringstream oss;
                oss << "<|blank_" << sm.str().size() << "|>";
                return oss.str();
                });
        }

        return output;
    }

    static inline std::string replace_punctuations(const std::string& text) {
        // reference: https://stackoverflow.com/questions/37989081/how-to-use-unicode-range-in-c-regex
        static std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
        static const std::vector<std::pair<std::wregex, std::wstring>> punct_map{
            {std::wregex(converter.from_bytes(R"(([\u4e00-\u9fff]),)")), converter.from_bytes("$1，")},
            {std::wregex(converter.from_bytes(R"(,([\u4e00-\u9fff]))")), converter.from_bytes("，$1")},
            {std::wregex(converter.from_bytes(R"(([\u4e00-\u9fff])!)")), converter.from_bytes("$1！")},
            {std::wregex(converter.from_bytes(R"(!([\u4e00-\u9fff]))")), converter.from_bytes("！$1")},
            {std::wregex(converter.from_bytes(R"(([\u4e00-\u9fff]):)")), converter.from_bytes("$1：")},
            {std::wregex(converter.from_bytes(R"(:([\u4e00-\u9fff]))")), converter.from_bytes("：$1")},
            {std::wregex(converter.from_bytes(R"(([\u4e00-\u9fff]);)")), converter.from_bytes("$1；")},
            {std::wregex(converter.from_bytes(R"(;([\u4e00-\u9fff]))")), converter.from_bytes("；$1")},
            {std::wregex(converter.from_bytes(R"(([\u4e00-\u9fff])\?)")), converter.from_bytes("$1？")},
            {std::wregex(converter.from_bytes(R"(\?([\u4e00-\u9fff]))")), converter.from_bytes("？$1")},
        };
        std::wstring w_output = converter.from_bytes(text);
        for (const auto& punct_pair : punct_map) {
            w_output = std::regex_replace(w_output, punct_pair.first, punct_pair.second);
        }
        std::string output = converter.to_bytes(w_output);
        return output;
    }

    std::string ChatGLMTokenizer::postprocess(const std::string& text) {
        std::string output;

        // newline token
        {
            static const std::regex pattern(R"(<n>)");
            output = std::regex_replace(text, pattern, "\n");
        }
        // tab token
        {
            static const std::regex pattern(R"(<\|tab\|>)");
            output = std::regex_replace(output, pattern, "\t");
        }
        // blank tokens
        {
            static const std::regex pattern(R"(<\|blank_(\d+)\|>)");
            output = regex_replace(output, pattern,
                [](const std::smatch& sm) { return std::string(std::stoi(sm[1].str()), ' '); });
        }
        // punctuations
        output = replace_punctuations(output);

        return output;
    }

    // ===== ChatGLM2-6B =====

    ChatGLM2Tokenizer::ChatGLM2Tokenizer(std::string_view serialized_model_proto) {
        const auto status = sp.LoadFromSerializedProto(serialized_model_proto);
        CHATGLM_CHECK(status.ok()) << status.ToString();

        int special_id = sp.GetPieceSize();
        mask_token_id = special_id++;
        gmask_token_id = special_id++;
        smask_token_id = special_id++;
        sop_token_id = special_id++;
        eop_token_id = special_id++;
    }

    std::vector<int> ChatGLM2Tokenizer::encode(const std::string& text, int max_length) const {
        std::vector<int> ids;
        sp.Encode(text, &ids);
        ids.insert(ids.begin(), { gmask_token_id, sop_token_id }); // special prefix
        if ((int)ids.size() > max_length) {
            // sliding window: drop the least recent history while keeping the two special prefix tokens
            int num_drop = (int)ids.size() - max_length;
            ids.erase(ids.begin() + 2, ids.begin() + 2 + num_drop);
        }
        return ids;
    }

    std::string ChatGLM2Tokenizer::decode(const std::vector<int>& ids) const {
        // filter out special tokens
        std::vector<int> normal_ids(ids);
        normal_ids.erase(std::remove_if(normal_ids.begin(), normal_ids.end(), [this](int id) { return is_special_id(id); }),
            normal_ids.end());

        std::string text;
        sp.Decode(normal_ids, &text);
        text = replace_punctuations(text);
        return text;
    }

    std::vector<int> ChatGLM2Tokenizer::encode_history(const std::vector<std::string>& history, int max_length) const {
        std::string prompt = build_prompt(history);
        std::vector<int> input_ids = encode(prompt, max_length);
        return input_ids;
    }

    std::string ChatGLM2Tokenizer::build_prompt(const std::vector<std::string>& history) {
        CHATGLM_CHECK(history.size() % 2 == 1) << "invalid history size " << history.size();

        std::ostringstream oss_prompt;
        for (size_t i = 0; i < history.size(); i += 2) {
            oss_prompt << "[Round " << i / 2 + 1 << "]\n\n问：" << history[i] << "\n\n答：";
            if (i < history.size() - 1) {
                oss_prompt << history[i + 1] << "\n\n";
            }
        }
        return oss_prompt.str();
    }

    bool ChatGLM2Tokenizer::is_special_id(int id) const {
        return id == mask_token_id || id == gmask_token_id || id == smask_token_id || id == sop_token_id ||
            id == eop_token_id;
    }

    // ===== ChatGLM3-6B =====

    ChatGLM3Tokenizer::ChatGLM3Tokenizer(std::string_view serialized_model_proto) {
        const auto status = sp.LoadFromSerializedProto(serialized_model_proto);
        CHATGLM_CHECK(status.ok()) << status.ToString();

        int special_id = sp.GetPieceSize();
        mask_token_id = special_id++;
        gmask_token_id = special_id++;
        smask_token_id = special_id++;
        sop_token_id = special_id++;
        eop_token_id = special_id++;
        system_token_id = special_id++;
        user_token_id = special_id++;
        assistant_token_id = special_id++;
        observation_token_id = special_id++;

        special_tokens = {
            {"[MASK]", mask_token_id},
            {"[gMASK]", gmask_token_id},
            {"[sMASK]", smask_token_id},
            {"sop", sop_token_id},
            {"eop", eop_token_id},
            {"<|system|>", system_token_id},
            {"<|user|>", user_token_id},
            {"<|assistant|>", assistant_token_id},
            {"<|observation|>", observation_token_id},
        };

        for (const auto& item : special_tokens) {
            index_special_tokens[item.second] = item.first;
        }
    }

    std::vector<int> ChatGLM3Tokenizer::encode(const std::string& text, int max_length) const {
        std::vector<int> ids;
        sp.Encode(text, &ids);
        ids.insert(ids.begin(), { gmask_token_id, sop_token_id }); // special prefix
        truncate(ids, max_length);
        return ids;
    }

    std::string ChatGLM3Tokenizer::decode(const std::vector<int>& ids) const {
        std::string text = decode_with_special_tokens(ids);
        text = remove_special_tokens(text);
        return text;
    }

    int ChatGLM3Tokenizer::get_command(const std::string& token) const {
        auto pos = special_tokens.find(token);
        CHATGLM_CHECK(pos != special_tokens.end()) << token << " is not a special token";
        return pos->second;
    }

    std::string ChatGLM3Tokenizer::decode_with_special_tokens(const std::vector<int>& ids) const {
        std::vector<std::string> pieces;
        for (int id : ids) {
            auto pos = index_special_tokens.find(id);
            if (pos != index_special_tokens.end()) {
                // special tokens
                pieces.emplace_back(pos->second);
            }
            else {
                // normal tokens
                pieces.emplace_back(sp.IdToPiece(id));
            }
        }

        std::string text = sp.DecodePieces(pieces);
        return text;
    }

    std::string ChatGLM3Tokenizer::remove_special_tokens(const std::string& text) {
        std::string output = text;
        static const std::vector<std::regex> special_token_regex{
            // std::regex(R"(<\|assistant\|> interpreter)"),
            // std::regex(R"(<\|assistant\|> interpre)"),
            std::regex(R"(<\|assistant\|>)"),
            std::regex(R"(<\|user\|>)"),
            std::regex(R"(<\|observation\|>)"),
        };
        for (const auto& re : special_token_regex) {
            output = std::regex_replace(output, re, "");
        }
        return output;
    }

    std::vector<int> ChatGLM3Tokenizer::encode_history(const std::vector<std::string>& history, int max_length) const {
        std::vector<int> input_ids{ gmask_token_id, sop_token_id };
        for (size_t i = 0; i < history.size(); i++) {
            std::string role = "user";
            std::vector<int> msg_ids;
            msg_ids.emplace_back(get_command("<|" + role + "|>"));
            // TODO: support metadata
            std::vector<int> newline_ids;
            sp.Encode("\n", &newline_ids);
            msg_ids.insert(msg_ids.end(), newline_ids.begin(), newline_ids.end());
            std::vector<int> content_ids;
            sp.Encode(history[i], &content_ids);
            msg_ids.insert(msg_ids.end(), content_ids.begin(), content_ids.end());

            input_ids.insert(input_ids.end(), msg_ids.begin(), msg_ids.end());

        }
        input_ids.emplace_back(assistant_token_id);
        truncate(input_ids, max_length);
        return input_ids;
    }

    bool ChatGLM3Tokenizer::is_special_id(int id) const {
        return id == mask_token_id || id == gmask_token_id || id == smask_token_id || id == sop_token_id ||
            id == eop_token_id || id == system_token_id || id == user_token_id || id == assistant_token_id ||
            id == observation_token_id;
    }

    void ChatGLM3Tokenizer::truncate(std::vector<int>& ids, int max_length) {
        if ((int)ids.size() > max_length) {
            // sliding window: drop the least recent history while keeping the two special prefix tokens
            int num_drop = (int)ids.size() - max_length;
            ids.erase(ids.begin() + 2, ids.begin() + 2 + num_drop);
        }
    }



#ifdef _POSIX_MAPPED_FILES
    MappedFile::MappedFile(const std::string& path) {
        int fd = open(path.c_str(), O_RDONLY);
        CHATGLM_CHECK(fd > 0) << "cannot open file " << path << ": " << strerror(errno);

        struct stat sb;
        CHATGLM_CHECK(fstat(fd, &sb) == 0) << strerror(errno);
        size = sb.st_size;

        data = (char*)mmap(nullptr, size, PROT_READ, MAP_SHARED, fd, 0);
        CHATGLM_CHECK(data != MAP_FAILED) << strerror(errno);

        CHATGLM_CHECK(close(fd) == 0) << strerror(errno);
    }

    MappedFile::~MappedFile() { CHATGLM_CHECK(munmap(data, size) == 0) << strerror(errno); }
#elif defined(_WIN32)
    MappedFile::MappedFile(const std::string& path) {

        int fd = open(path.c_str(), O_RDONLY);
        CHATGLM_CHECK(fd > 0) << "cannot open file " << path << ": " << strerror(errno);

        struct _stat64 sb;
        CHATGLM_CHECK(_fstat64(fd, &sb) == 0) << strerror(errno);
        size = sb.st_size;

        HANDLE hFile = (HANDLE)_get_osfhandle(fd);

        HANDLE hMapping = CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
        CHATGLM_CHECK(hMapping != NULL) << strerror(errno);

        data = (char*)MapViewOfFile(hMapping, FILE_MAP_READ, 0, 0, 0);
        CloseHandle(hMapping);

        CHATGLM_CHECK(data != NULL) << strerror(errno);

        CHATGLM_CHECK(close(fd) == 0) << strerror(errno);
    }

    MappedFile::~MappedFile() { CHATGLM_CHECK(UnmapViewOfFile(data)) << strerror(errno); }
#endif

    void ModelLoader::seek(int64_t offset, int whence) {
        if (whence == SEEK_SET) {
            ptr = data + offset;
        }
        else if (whence == SEEK_CUR) {
            ptr += offset;
        }
        else if (whence == SEEK_END) {
            ptr = data + size + offset;
        }
        else {
            CHATGLM_THROW << "invalid seek mode " << whence;
        }
    }

    std::string ModelLoader::read_string(size_t length) {
        std::string s(ptr, ptr + length);
        ptr += length;
        return s;
    }

    // ===== pipeline =====

    Pipeline::Pipeline(const std::string& path) {
        mapped_file = std::make_unique<MappedFile>(path);
        ModelLoader loader(mapped_file->data, mapped_file->size);

        // load magic
        std::string magic = loader.read_string(4);
        CHATGLM_CHECK(magic == "ggml") << "model file is broken (bad magic)";

        // load model type
        ModelType model_type = (ModelType)loader.read_basic<int>();
        // load version
        int version = loader.read_basic<int>();
        if (model_type == MODEL_TYPE_CHATGLM) {
            CHATGLM_CHECK(version == 1) << "only support version 1 for now but got " << version;

            // load tokenizer
            int proto_size = loader.read_basic<int>();
            std::string_view serialized_model_proto((char*)mapped_file->data + loader.tell(), proto_size);
            loader.seek(proto_size, SEEK_CUR);
            tokenizer = std::make_unique<ChatGLMTokenizer>(serialized_model_proto);
        }
        else if (model_type == MODEL_TYPE_CHATGLM2 || model_type == MODEL_TYPE_CHATGLM3) {
            CHATGLM_CHECK(version == 1) << "only support version 1 for now but got " << version;

            // load tokenizer
            int proto_size = loader.read_basic<int>();
            std::string_view serialized_model_proto((char*)mapped_file->data + loader.tell(), proto_size);
            loader.seek(proto_size, SEEK_CUR);


            if (model_type == MODEL_TYPE_CHATGLM2) {
                tokenizer = std::make_unique<ChatGLM2Tokenizer>(serialized_model_proto);
            }
            else {
                tokenizer = std::make_unique<ChatGLM3Tokenizer>(serialized_model_proto);
            }
        }
        else {
            CHATGLM_THROW << "invalid model type " << model_type;
        }
    }

    // ===== Tokenizer =====

    Tokenizer::Tokenizer(const std::string& path) {
        mapped_file = std::make_unique<MappedFile>(path);
        ModelLoader loader(mapped_file->data, mapped_file->size);

        // load magic
        std::string magic = loader.read_string(4);
        CHATGLM_CHECK(magic == "ggml") << "model file is broken (bad magic)";

        // load model type
        ModelType model_type = (ModelType)loader.read_basic<int>();
        // load version
        int version = loader.read_basic<int>();
        if (model_type == MODEL_TYPE_CHATGLM) {
            CHATGLM_CHECK(version == 1) << "only support version 1 for now but got " << version;

            // load tokenizer
            int proto_size = loader.read_basic<int>();
            std::string_view serialized_model_proto((char*)mapped_file->data + loader.tell(), proto_size);
            loader.seek(proto_size, SEEK_CUR);
            tokenizer = std::make_unique<ChatGLMTokenizer>(serialized_model_proto);
        }
        else if (model_type == MODEL_TYPE_CHATGLM2 || model_type == MODEL_TYPE_CHATGLM3) {
            CHATGLM_CHECK(version == 1) << "only support version 1 for now but got " << version;

            // load tokenizer
            int proto_size = loader.read_basic<int>();
            std::string_view serialized_model_proto((char*)mapped_file->data + loader.tell(), proto_size);
            loader.seek(proto_size, SEEK_CUR);


            if (model_type == MODEL_TYPE_CHATGLM2) {
                tokenizer = std::make_unique<ChatGLM2Tokenizer>(serialized_model_proto);
            }
            else {
                tokenizer = std::make_unique<ChatGLM3Tokenizer>(serialized_model_proto);
            }

            std::vector<int> input_ids = tokenizer->encode("what is openvino", 1024);
            std::string output = tokenizer->decode(input_ids);
        }
        else {
            CHATGLM_THROW << "invalid model type " << model_type;
        }
    }

    std::vector<int> Tokenizer::encode(const std::string& prompt, int max_context_length) const {
        std::vector<int> input_ids = tokenizer->encode(prompt, max_context_length);
        return input_ids;
    }

    std::string Tokenizer::decode(const std::vector<int>& input_ids) const {
        std::string output = tokenizer->decode(input_ids);
        return output;
    }

    std::vector<int> Tokenizer::encode_history(const std::vector<std::string>& history, int max_length) const {
        std::vector<int> input_ids = tokenizer->encode_history(history, max_length);
        return input_ids;
    }

    void TextStreamer::put(const std::vector<int>& output_ids) {
        if (is_prompt_) {
            // skip prompt
            is_prompt_ = false;
            return;
        }

        static const std::vector<char> puncts{ ',', '!', ':', ';', '?' };

        token_cache_.insert(token_cache_.end(), output_ids.begin(), output_ids.end());
        std::string text = tokenizer_->decode(token_cache_);
        if (text.empty()) {
            return;
        }

        std::string printable_text;
        if (text.back() == '\n') {
            // flush the cache after newline
            printable_text = text.substr(print_len_);
            token_cache_.clear();
            print_len_ = 0;
        }
        else if (std::find(puncts.begin(), puncts.end(), text.back()) != puncts.end()) {
            // last symbol is a punctuation, hold on
        }
        else if (text.size() >= 3 && text.compare(text.size() - 3, 3, "�") == 0) {
            // ends with an incomplete token, hold on
        }
        else {
            printable_text = text.substr(print_len_);
            print_len_ = text.size();
        }

        os_ << printable_text << std::flush;
    }

    void TextStreamer::end() {
        std::string text = tokenizer_->decode(token_cache_);
        os_ << text.substr(print_len_) << std::endl;
        is_prompt_ = true;
        token_cache_.clear();
        print_len_ = 0;
    }
}
