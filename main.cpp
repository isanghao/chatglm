


#include <iomanip>
#include <iostream>
#include <sstream>
#include <unordered_map>
#include <vector>
#include "chatglm.h"

#include <openvino/openvino.hpp>

#ifdef _WIN32
#include <codecvt>
#include <fcntl.h>
#include <io.h>
#include <windows.h>
#endif

typedef std::chrono::high_resolution_clock Time;
typedef std::chrono::nanoseconds ns;

const std::string sentences[] =
{
    "What is OpenVINO?",
    "If I have 100 million dollars, what kinds of projects should I invest to maximize my benefits in background of a growing number of artificial intelligence technologies?",
    "Originally, There were three types of cake in the cake store: Strawberry Cream Cake, Chocolate Coconut Cake, and Red Velvet Brownie Cake. Customer number is large enough so that no cake would be left every day when the store close. As the name suggested, each cake has two ingredients: Strawberry Cream Cake with strawberries and cream, Chocolate Coconut Cake with chocolate and coconut, and Red Velvet Brownie Cake with red velvet and brownie. Different ingredients can be compatibly mixed with each other without any issue. After the cake is made, there are often some leftover materials for each ingredient. In order to reduce waste, the store often combine the extra ingredients in pairs to make new small gifts to gain extra sales. For example, strawberries and chocolate can be mixed to create strawberry-flavored chocolate sauce, and brownies and shredded coconut can be mixed to create brownie coconut cookies. Only two ingredients can be mixed, and mixture with more than two ingredients can cost a lot of time and will not be adopted. In order to decrease the problem complexity, the store will also prevent from careful decorations or other difficult steps as in procedure of making cakes, so that time cost can be omited. By analogy, if all the ingredients can be combined in pairs, what small products can the store make in the end?",
    "There is a table, which contains three drawers: left drawer, middle drawer and right drawer; Tom Ethan, Elbert Alex, Jack Johnson, and Mario Thompson all saw a bag of chocolates on the table. Tom Ethan asked Elbert Alex and Jack Johnson to go out, and after that, he put the bag of chocolates in the right drawer in front of Mario Thompson; after Jack Johnson came back, Tom Ethan asked Mario Thompson to go out to find Elbert Alex, and took it from the left drawer in front of Jack Johnson. Then He take out a box of biscuits and put them in the middle drawer; when Elbert Alex and Mario Thompson returned, Tom Ethan asked Jack Johnson and Mario Thompson to go out to buy a bottle of soy sauce. Tom Ethan waited for a long time, and found that Jack Johnson and Mario Thompson had not returned, so he sent Elbert Alex to look for them, but in the end only Jack Johnson and Elbert Alex came back. Jack Johnson told Tom Ethan that at first they could not find any shop that is providing soy sauce, so they had to separate to search other shops, which is why Mario Thompson got lost; on the way back, Jack Johnson ran into Elbert Alex, and they rushed back first. Therefore, Tom Ethan asked them to go out to find Mario Thompson again; in order to prevent getting lost again, Tom Ethan told Elbert Alex and Jack Johnson to walk together at all time, and even if they could not get the soy sauce, they had to find and get back with Mario Thompson. As a result, Elbert Alex and Jack Johnson found Mario Thompson outside and found that he had bought a bottle of soy sauce. The three felt that Tom Ethan never went out to do anthing but they are busy all the time. So they were very angry. They discussed and made a conclusion. After going back to see Tom Ethan, they should not tell him about the soy sauce they bought, and asked Jack Johnson to hide the soy sauce in his backpack. After the three of them came back together, they pretended to claim that they did not foudn and bought soy sauce according to the plan, and hoped that Tom Ethan would go out together to buy things in the future, and he should not be so lazy. Tom Ethan agreed and felt sory about that. When everyone finally stood in front of the table, the four of them wrote down the list of items they knew and the location of the items. So the question is: is the information writen by these four people consistent, and why?",
    "The process of Origami seems simple at the first glance, but in fact, it still requires a very complicated process to do it well. Taking folding a rose as an example, we can divide the entire process into three stages, including: firstly creating a grid of creases, secondly making a three-dimensional base, and thirdly finishing petal decoration. The first step is to create a grid of creases: this step is a bit like the first step of folding a gift of thousand-paper-crane. That is to say, we can fold the paper in half (or namedly equal-folds) through the symmetrical axis, and repeat such step in the other symmetrical axis. And then apply multiple equal-folds in sequence relative to each smaller rectangle divided by the two creases; After that, the creases in each direction will interweave into a complete set of uniform small square splicing patterns; these small squares form a reference space similar to a two-dimensional coordinate system, allowing us to combine adjacent creases on the plane from Three-dimensional high platforms or depressions are folded on the two-dimensional small squares to facilitate the next steps of folding. It should be noted that, in the process of creating grid creases, there may be rare cases when the folds are not aligned. The consequences of this error can be very serious. And just like the butterfly effect, it is only a slight difference at the beginning , and in the end it may generate a disaster world which is completely different from plan. Anyway, let's continue. The second step is make the three-dimensional base: In this step, we need to fold a set of symmetrical three-dimensional high platforms or depressions based on the grid creases. From the symmetry analysis, it is not difficult to find that the rose will have four symmetrical three-dimensional high platforms and supporting depressions. Therefore, we can firstly fold out a quarter of the depression and plateau patterns, which would help build a base to compose into a complex 3D structure. And then, we use this quarter as a template, and fold out the repeating patterns on the remaining three parts of the whole structure in turn. It is worth noting that the layout of the high platform not only needs to consider the regular contrast and symmetrical distribution of the length and width, but also needs to ensure the orderliness of the height dimension. This is very important, since we will never go back to this step after all parts were made, and you would better start from first step if you make anything wrong in the this step. Similar to the precautions in the first stage, please handle all the corners in three dimensions to ensure that they conform to the layout required in the plan, which would help us avoid the butterfly effect and increase the robustness in the process of three-dimensional folding. Just like building a skyscrapper in the real world, people usually take a lot of time when building the base but soon get finished when extending the structure after that. Time is worth to cost in the base, but would be saved in the future after you succeed in base. Anyway, let's continue. During the first quarter of the pattern, repeated comparisons with the finished rose were made to eliminate any possible errors in the first place. The final stage is to finish the petal grooming. At this stage, we often emphasize an important term called folding-by-heart. The intention here is no longer literally serious, but focus is moved to our understanding of the shape of a rose in nature, and we usually use natural curves to continuously correct the shape of petals in order to approach the shape of rose petals in reality. One more comment: this is also the cause of randomness to the art, which can be generated differently by different people. Recall that rose should be adjusted close to reality, so in the last step of this stage, we need to open the bloom in the center of the rose, by pulling on the four petals that have been bent. This process may be accompanied by the collapse of the overall structure of the rose, so we should be very careful to save strength of adjustment, and it must be well controlled to avoid irreversible consequences. Ultimately, after three stages of folding, we end up with a crown of rose with a similar shape close to reality. If condition is permited, we can wrap a green paper strip twisted on a straightened iron wire, and insert the rose crown we just created onto one side of the iron wire. In this way, we got a hand-made rose with a green stem. We can also repeat the steps above to increase the number of rose, so that it can be made into a cluster. Different color of rose is usually more attractive and can be considered as a better plan of gift to your friend. In summary, by creating a grid of creases, making a three-dimensional base, and finishing with petals, we created a three-dimensional rose from a two-dimensional paper. Although this process may seem simple, it is indeed a work of art created by us humans with the help of imagination and common materials. At last, Please comment to assess the above content.",
};

double get_duration_ms_until_now(Time::time_point& startTime) {
    return std::chrono::duration_cast<ns>(Time::now() - startTime).count() * 0.000001;
}

struct Args {
    std::string model_path = "chatglm_tokenizer.bin";
    std::string ov_model_path = "openvino_model.xml";
    std::string prompt = "你好";
    std::string device = "CPU";
    int max_length = 2048;
    int max_context_length = 512;
};

static void usage(const std::string& prog) {
    std::cout << "Usage: " << prog << " [options]\n"
        << "\n"
        << "options:\n"
        << "  -h, --help              show this help message and exit\n"
        << "  -m, --model PATH        model path (default: chatglm-ggml.bin)\n"
        << "  -m_ov, --ov_model PATH  ov model path (default: openvino_model.xml)\n"
        << "  -d, --device            inference device (default: CPU)\n"
        << "  -p, --prompt PROMPT     prompt to start generation with (default: 你好)\n"
        << "  -l, --max_length N      max total length including prompt and output (default: 2048)\n"
        << "  -c, --max_context_length N\n"
        << "                          max context length (default: 512)\n";
}

namespace {
    
    ov::Tensor glmpositionidsgenerator(int qlen, int n_past, int n_ctx) {
        auto tensor = ov::Tensor(ov::element::i32, { 1, (size_t)qlen });
        int32_t* position_ids = tensor.data<int32_t>();
        for (int i = 0; i < qlen; i++) {
            (position_ids)[i] = i + n_past;
        }

        return tensor;
    }
}

static Args parse_args(const std::vector<std::string>& argv) {
    Args args;

    for (size_t i = 1; i < argv.size(); i++) {
        const std::string& arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            usage(argv[0]);
            exit(EXIT_SUCCESS);
        }
        else if (arg == "-m" || arg == "--model") {
            args.model_path = argv[++i];
        }
        else if (arg == "-m_ov" || arg == "--ov_model") {
            args.ov_model_path = argv[++i];
        }
        else if (arg == "-d" || arg == "--device") {
            args.device = argv[++i];
        }
        else if (arg == "-p" || arg == "--prompt") {
            args.prompt = argv[++i];
        }
        else if (arg == "-l" || arg == "--max_length") {
            args.max_length = std::stoi(argv[++i]);
        }
        else if (arg == "-c" || arg == "--max_context_length") {
            args.max_context_length = std::stoi(argv[++i]);
        }
        else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            usage(argv[0]);
            exit(EXIT_FAILURE);
        }
    }

    return args;
}

static Args parse_args(int argc, char** argv) {
    std::vector<std::string> argv_vec;
    argv_vec.reserve(argc);

#ifdef _WIN32
    LPWSTR* wargs = CommandLineToArgvW(GetCommandLineW(), &argc);
    //CHATGLM_CHECK(wargs) << "failed to retrieve command line arguments";

    std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
    for (int i = 0; i < argc; i++) {
        argv_vec.emplace_back(converter.to_bytes(wargs[i]));
    }

    LocalFree(wargs);
#else
    for (int i = 0; i < argc; i++) {
        argv_vec.emplace_back(argv[i]);
    }
#endif

    return parse_args(argv_vec);
}

static bool get_utf8_line(std::string& line) {
#ifdef _WIN32
    std::wstring wline;
    bool ret = !!std::getline(std::wcin, wline);
    std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
    line = converter.to_bytes(wline);
    return ret;
#else
    return !!std::getline(std::cin, line);
#endif
}

#define COMPILE_FROM_XML 0

int main(int argc, char** argv) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    _setmode(_fileno(stdin), _O_WTEXT);
#endif

    try {
        Args args = parse_args(argc, argv);

        ov::Core core;
        core.set_property(ov::cache_dir(".\\cache"));

        //init chatglm tokenizer
        double total_time;
        auto startTime = Time::now();
        chatglm::Tokenizer tokenizer(args.model_path);
        auto duration_ms = get_duration_ms_until_now(startTime);
        std::cout << "load tokenizer took " << duration_ms << " ms" << std::endl;

        auto text_streamer = std::make_shared<chatglm::TextStreamer>(std::cout, tokenizer.tokenizer.get());

        std::vector<int> input_ids = tokenizer.encode_history({args.prompt}, args.max_context_length);
        std::string output = tokenizer.decode(input_ids);

        // std::this_thread::sleep_until(std::chrono::system_clock::now() + std::chrono::seconds(10));

        startTime = Time::now();
#if !COMPILE_FROM_XML
        std::shared_ptr<ov::Model> model = core.read_model(args.ov_model_path);
#endif
        duration_ms = get_duration_ms_until_now(startTime);
        std::cout << "read_model took " << duration_ms << " ms" << std::endl;

        // std::this_thread::sleep_until(std::chrono::system_clock::now() + std::chrono::seconds(10));


        constexpr size_t BATCH_SIZE = 1;

        int seqlen = input_ids.size();
        int npast = 0;
        int nctx = seqlen;
  
        auto  position_ids = glmpositionidsgenerator(seqlen, npast, nctx);

#if !COMPILE_FROM_XML        
        /**/
        //std::cout << "position ids shape " << position_ids.get_shape() << position_ids.get_size() << std::endl;

        std::map<std::string, ov::PartialShape> shapes = {
            {"input_ids", ov::PartialShape{
                BATCH_SIZE, {1, (std::numeric_limits<ov::Dimension::value_type>::max)()}
            }},
            {"attention_mask", ov::PartialShape{
                BATCH_SIZE, {1, (std::numeric_limits<ov::Dimension::value_type>::max)()}
            }},
            {"position_ids", ov::PartialShape{
                BATCH_SIZE, {1, (std::numeric_limits<ov::Dimension::value_type>::max)()}
            }}
        };

        for (const ov::Output<ov::Node>& input : model->inputs()) {
            for (const std::string& name : input.get_names()) {
                if (name.rfind("past_key_values", 0) == 0) {
                    ov::PartialShape shape = input.get_partial_shape();
                    shape[1] = BATCH_SIZE;
                    shapes.emplace(name, shape);
                    break;
                }
            }
        }
        
        model->reshape(shapes);
        {
            ov::preprocess::PrePostProcessor p3(model);
            p3.input("input_ids").tensor().set_element_type(ov::element::i32);  // cast to the type of tokenyzer's output
            p3.input("attention_mask").tensor().set_element_type(ov::element::i32);
            p3.input("position_ids").tensor().set_element_type(ov::element::i32);
            model = p3.build();
        }
#endif

        // std::cout << "Reshape" << std::endl;
        // std::this_thread::sleep_until(std::chrono::system_clock::now() + std::chrono::seconds(10));
        //ov::InferRequest ireq = core.compile_model(model, "CPU", { ov::cache_dir("llm-cache") }).create_infer_request();
        //ov::InferRequest ireq = core.compile_model(model, "CPU").create_infer_request();
        startTime = Time::now();
#if !COMPILE_FROM_XML
        ov::CompiledModel compilemodel = core.compile_model(model, args.device);
#else
        ov::CompiledModel compilemodel = core.compile_model(args.ov_model_path, args.device);
#endif
        duration_ms = get_duration_ms_until_now(startTime);
        std::cout << "Compile LLM model took " << duration_ms << " ms" << std::endl;

        ov::InferRequest ireq;
        int32_t out_token;

#if !COMPILE_FROM_XML
        auto model_inputs = model->inputs();
        model = nullptr;
#else
        auto model_inputs = compilemodel.inputs();
#endif
        std::cout << "Get inputs" << std::endl;
        // std::this_thread::sleep_until(std::chrono::system_clock::now() + std::chrono::seconds(10));

        for (std::string input_text : sentences) {
            std::cout << " #### sentence: index " << input_text << std::endl;

            startTime = Time::now();
            ireq = compilemodel.create_infer_request();
            duration_ms = get_duration_ms_until_now(startTime);
            std::cout << "compilemodel.create_infer_request took " << duration_ms << " ms" << std::endl;

            if (text_streamer) {
                text_streamer->put(input_ids);
            }

            startTime = Time::now();
            input_ids = tokenizer.encode_history({ input_text }, args.max_context_length);
            duration_ms = get_duration_ms_until_now(startTime);
            std::cout << "Get Tokenizer id  took " << duration_ms << " ms" << std::endl;

#if !COMPILE_FROM_XML
            for (const ov::Output<ov::Node>& input : model_inputs) {
#else
            for (auto& input : model_inputs) {
#endif
                for (const std::string& name : input.get_names()) {
                    if (name.rfind("past_key_values", 0) == 0) {
                        ireq.get_tensor(input).set_shape(input.get_partial_shape().get_min_shape());
                        //std::cout << "name " << name << "shape" << input.get_partial_shape().get_min_shape() << std::endl;
                        break;
                    }
                }
            }

            startTime = Time::now();
            seqlen = input_ids.size();
            npast = 0;
            nctx = seqlen;
            position_ids = glmpositionidsgenerator(seqlen, npast, nctx);
            ireq.get_tensor("input_ids").set_shape({ BATCH_SIZE, input_ids.size() });  // TODO: replace with ireq.set_tensor("input_ids", input_ids); after it's fixed
            ireq.get_tensor("attention_mask").set_shape({ BATCH_SIZE, input_ids.size() });
            ireq.get_tensor("position_ids").set_shape(position_ids.get_shape());
            std::copy_n(input_ids.data(), input_ids.size(), ireq.get_tensor("input_ids").data<int32_t>());
            std::fill_n(ireq.get_tensor("attention_mask").data<int32_t>(), input_ids.size(), 1);
            std::copy_n(position_ids.data<const int32_t>(), position_ids.get_size(), ireq.get_tensor("position_ids").data<int32_t>());
            duration_ms = get_duration_ms_until_now(startTime);
            std::cout << "input toker length " << seqlen << " Copy tokernizer " << duration_ms << " ms" << std::endl;

            startTime = Time::now();
            ireq.infer();
            duration_ms = get_duration_ms_until_now(startTime);
            std::cout << "First inference took " << duration_ms << " ms" << std::endl;

            size_t vocab_size = ireq.get_tensor("logits").get_shape().back();
            float* logits = ireq.get_tensor("logits").data<float>() + (input_ids.size() - 1) * vocab_size;
            out_token = int32_t(std::max_element(logits, logits + vocab_size) - logits);
            
            if (text_streamer) {
                text_streamer->put({ out_token });
            }

            ireq.get_tensor("input_ids").set_shape({ BATCH_SIZE, 1 });
            ireq.get_tensor("attention_mask").set_shape({ BATCH_SIZE, 1 });
            ireq.get_tensor("attention_mask").data<int32_t>()[0] = 1;
            npast = nctx;
            position_ids = glmpositionidsgenerator(1, npast, nctx);
            ireq.get_tensor("position_ids").set_shape(position_ids.get_shape());

            total_time = 0;
            int count = 0;
            double second_time = 0;

            constexpr int32_t SPECIAL_EOS_TOKEN = 2;  // There's no way to extract the value from the tokenizer for now
            while (out_token != SPECIAL_EOS_TOKEN && count < 50) {
                startTime = Time::now();
#if !COMPILE_FROM_XML
                for (const ov::Output<ov::Node>& input : model_inputs) {
#else
                for (auto& input : model_inputs) {
#endif
                    for (const std::string& name : input.get_names()) {
                        if (name.rfind("past_key_values", 0) == 0) {
                            ireq.set_tensor(input, ireq.get_tensor("present" + name.substr(15)));
                            break;
                        }
                    }
                }
                position_ids = glmpositionidsgenerator(1, npast, nctx);
                ireq.get_tensor("position_ids").set_shape(position_ids.get_shape());
                std::copy_n(position_ids.data<const int32_t>(), position_ids.get_size(), ireq.get_tensor("position_ids").data<int32_t>());
                npast = npast + 1;

                ireq.get_tensor("input_ids").data<int32_t>()[0] = out_token;
                auto inferStartTime = Time::now();
                ireq.start_async();
                //print_token(tokenizer, out_token);
                ireq.wait();
                auto infer_duration_ms = std::chrono::duration_cast<ns>(Time::now() - inferStartTime).count() * 0.000001;
                std::cout << "infer " << infer_duration_ms << "ms" << std::endl;
                duration_ms = get_duration_ms_until_now(startTime);
                count += 1;

                logits = ireq.get_tensor("logits").data<float>();
                out_token = int32_t(std::max_element(logits, logits + vocab_size) - logits);
                out_token = 1;

                //std::cout << " out_token " << out_token << std::endl;

                if (text_streamer) {
                    text_streamer->put({ out_token });
                }/**/

                if (count != 1) {
                    total_time += duration_ms;
                }
                else {
                    second_time = duration_ms;
                }

                if (count + 1 > args.max_context_length) {
                    break;
                }
            }

            if (text_streamer) {
                text_streamer->end();
            }

            std::cout << '\n';
            std::cout << "Second inference took " << second_time << " ms" << std::endl;
            if (count > 2) {
                std::cout << "Other Avg inference took total " << total_time << " ms token num " << count - 1 << " avg " << total_time / (count - 1) << " ms" << std::endl;
            }
            break;

        }
    }
    catch (std::exception& e) {
        std::cerr << e.what() << std::endl;
        exit(EXIT_FAILURE);
    }
    return 0;
}
