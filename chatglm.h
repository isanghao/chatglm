
#include <cmath>
#include <sstream>
#include <unordered_map>
#include <vector>
#include <sentencepiece_processor.h>

namespace chatglm {

    // ===== common =====

    static constexpr size_t MB = 1024 * 1024;

    class LogMessageFatal {
    public:
        LogMessageFatal(const char* file, int line) { oss_ << file << ':' << line << ' '; }
        [[noreturn]] ~LogMessageFatal() noexcept(false) { throw std::runtime_error(oss_.str()); }
        std::ostringstream& stream() { return oss_; }

    private:
        std::ostringstream oss_;
    };

#define CHATGLM_THROW ::chatglm::LogMessageFatal(__FILE__, __LINE__).stream()
#define CHATGLM_CHECK(cond)                                                                                            \
    if (!(cond))                                                                                                       \
    CHATGLM_THROW << "check failed (" #cond ") "

    enum ModelType {
        MODEL_TYPE_CHATGLM = 1,
        MODEL_TYPE_CHATGLM2 = 2,
        MODEL_TYPE_CHATGLM3 = 3,
        MODEL_TYPE_BAICHUAN7B = 1024,
        MODEL_TYPE_BAICHUAN13B = 1025,
        MODEL_TYPE_INTERNLM = 1280,
    };

    class BaseTokenizer {
    public:
        virtual ~BaseTokenizer() = default;
        virtual std::vector<int> encode(const std::string& text, int max_length) const = 0;
        virtual std::string decode(const std::vector<int>& ids) const = 0;
        virtual std::vector<int> encode_history(const std::vector<std::string>& history, int max_length) const = 0;
    };

    // ===== ChatGLM-6B =====

    class ChatGLMTokenizer : public BaseTokenizer {
    public:
        ChatGLMTokenizer(std::string_view serialized_model_proto);

        std::vector<int> encode(const std::string& text, int max_length) const override;

        std::string decode(const std::vector<int>& ids) const override;

        std::vector<int> encode_history(const std::vector<std::string>& history, int max_length) const override;

        static std::string build_prompt(const std::vector<std::string>& history);

    private:
        static std::string preprocess(const std::string& text);

        static std::string postprocess(const std::string& text);

    public:
        sentencepiece::SentencePieceProcessor sp;
        int bos_token_id;
        int eos_token_id;
        int mask_token_id;
        int gmask_token_id;
        int pad_token_id;
    };


    // ===== ChatGLM2-6B =====

    class ChatGLM2Tokenizer : public BaseTokenizer {
    public:
        ChatGLM2Tokenizer(std::string_view serialized_model_proto);

        std::vector<int> encode(const std::string& text, int max_length) const override;

        std::string decode(const std::vector<int>& ids) const override;

        std::vector<int> encode_history(const std::vector<std::string>& history, int max_length) const override;

        static std::string build_prompt(const std::vector<std::string>& history);

        bool is_special_id(int id) const;

    public:
        sentencepiece::SentencePieceProcessor sp;
        int mask_token_id;
        int gmask_token_id;
        int smask_token_id;
        int sop_token_id;
        int eop_token_id;
    };

    // ===== ChatGLM3-6B =====

    class ChatGLM3Tokenizer : public BaseTokenizer {
    public:
        ChatGLM3Tokenizer(std::string_view serialized_model_proto);

        std::vector<int> encode(const std::string& text, int max_length) const override;

        std::string decode(const std::vector<int>& ids) const override;

        std::vector<int> encode_history(const std::vector<std::string>& history, int max_length) const override;

        bool is_special_id(int id) const;

    private:
        static void truncate(std::vector<int>& ids, int max_length);

        std::string decode_with_special_tokens(const std::vector<int>& ids) const;

        static std::string remove_special_tokens(const std::string& text);

        int get_command(const std::string& token) const;

    public:
        sentencepiece::SentencePieceProcessor sp;
        int mask_token_id;
        int gmask_token_id;
        int smask_token_id;
        int sop_token_id;
        int eop_token_id;
        int system_token_id;
        int user_token_id;
        int assistant_token_id;
        int observation_token_id;
        std::unordered_map<std::string, int> special_tokens;
        std::unordered_map<int, std::string> index_special_tokens;
    };


    class MappedFile {
    public:
        MappedFile(const std::string& path);
        ~MappedFile();

    public:
        char* data;
        size_t size;
    };

    class ModelLoader {
    public:
        ModelLoader(char* data, size_t size) : data(data), size(size), ptr(data) {}

        int64_t tell() const { return ptr - data; }

        void seek(int64_t offset, int whence);

        template <typename T>
        T read_basic() {
            T obj = *(T*)ptr;
            ptr += sizeof(T);
            return obj;
        }

        std::string read_string(size_t length);

    public:
        char* data;
        size_t size;
        char* ptr;
    };

    // ===== pipeline =====

    class Pipeline {
    public:
        Pipeline(const std::string& path);


    public:
        std::unique_ptr<BaseTokenizer> tokenizer;
        std::unique_ptr<MappedFile> mapped_file;
    };

    // ===== tokernizer ====
    class Tokenizer {
    public:
        Tokenizer(const std::string& path);

        std::vector<int> encode(const std::string& prompt, int max_context_length) const;

        std::string decode(const std::vector<int>& input_ids) const;

        std::vector<int> encode_history(const std::vector<std::string>& history, int max_length) const;

    public:
        std::unique_ptr<BaseTokenizer> tokenizer;
        std::unique_ptr<MappedFile> mapped_file;
    };

    class BaseStreamer {
    public:
        virtual ~BaseStreamer() = default;
        virtual void put(const std::vector<int>& output_ids) = 0;
        virtual void end() = 0;
    };

    // reference: https://github.com/huggingface/transformers/blob/main/src/transformers/generation/streamers.py
    class TextStreamer : public BaseStreamer {
    public:
        TextStreamer(std::ostream& os, BaseTokenizer* tokenizer)
            : os_(os), tokenizer_(tokenizer), is_prompt_(true), print_len_(0) {}
        void put(const std::vector<int>& output_ids) override;
        void end() override;

    private:
        std::ostream& os_;
        BaseTokenizer* tokenizer_;
        bool is_prompt_;
        std::vector<int> token_cache_;
        int print_len_;
    };
}

