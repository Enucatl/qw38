// Score exact frozen token IDs with llama.cpp's GPU runtime.
// Adapted from DS4 gguf-tools/quality-testing/score_llama.cpp.
#include "ggml-backend.h"
#include "ggml.h"
#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

static void fail(const std::string &message) {
    std::cerr << message << '\n';
    std::exit(1);
}

static std::vector<std::string> fields(const std::string &line) {
    std::vector<std::string> result;
    std::stringstream stream(line);
    std::string field;
    while (std::getline(stream, field, '\t')) result.push_back(field);
    return result;
}

static std::vector<llama_token> read_tokens(const std::string &path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) fail("cannot open " + path);
    std::vector<llama_token> tokens(512);
    for (auto &token : tokens) {
        unsigned char bytes[4];
        if (!file.read(reinterpret_cast<char *>(bytes), 4)) fail("short token window " + path);
        token = static_cast<llama_token>(uint32_t(bytes[0]) | uint32_t(bytes[1]) << 8 |
                                         uint32_t(bytes[2]) << 16 | uint32_t(bytes[3]) << 24);
    }
    if (file.get() != EOF) fail("long token window " + path);
    return tokens;
}

static void decode(llama_context *ctx, llama_batch &batch, const llama_token *tokens,
                   int count, int position, bool logits) {
    batch.n_tokens = count;
    for (int i = 0; i < count; ++i) {
        batch.token[i] = tokens[i];
        batch.pos[i] = position + i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = logits && i == count - 1;
    }
    if (llama_decode(ctx, batch) != 0) fail("llama_decode failed");
}

static double negative_log_probability(const float *logits, int vocab_size,
                                       llama_token target, llama_token &greedy) {
    if (target < 0 || target >= vocab_size) fail("target token outside model vocabulary");
    float maximum = -std::numeric_limits<float>::infinity();
    greedy = 0;
    for (int i = 0; i < vocab_size; ++i) {
        if (logits[i] > maximum) {
            maximum = logits[i];
            greedy = i;
        }
    }
    if (!std::isfinite(maximum)) fail("nonfinite logits");
    double total = 0.0;
    for (int i = 0; i < vocab_size; ++i) total += std::exp(double(logits[i] - maximum));
    return double(maximum) + std::log(total) - double(logits[target]);
}

struct Trace {
    std::string directory;
    std::string phase;
};

static bool trace_input(ggml_tensor *tensor, bool ask, void *user_data) {
    auto &trace = *static_cast<Trace *>(user_data);
    if (tensor->op != GGML_OP_MUL_MAT || !tensor->src[0] || !tensor->src[1]) return false;
    const std::string weight(tensor->src[0]->name);
    if (weight != "output.weight" && weight.find("blk.0.") != 0 &&
        weight.find("blk.31.") != 0 && weight.find("blk.63.") != 0) return false;
    if (weight != "output.weight" && weight.find("ffn_gate.weight") == std::string::npos &&
        weight.find("ffn_down.weight") == std::string::npos &&
        weight.find("attn_qkv.weight") == std::string::npos &&
        weight.find("attn_q.weight") == std::string::npos &&
        weight.find("attn_output.weight") == std::string::npos &&
        weight.find("ssm_out.weight") == std::string::npos) return false;
    const auto path = std::filesystem::path(trace.directory) / (weight + "." + trace.phase + ".bin");
    if (std::filesystem::exists(path)) return false;
    if (ask) return true;
    auto *input = tensor->src[1];
    if (!input->buffer) fail("trace input has no backend buffer: " + weight);
    std::vector<char> data(ggml_nbytes(input));
    ggml_backend_tensor_get(input, data.data(), 0, data.size());
    std::ofstream output(path, std::ios::binary);
    output.write(data.data(), static_cast<std::streamsize>(data.size()));
    if (!output) fail("failed to write trace: " + path.string());
    std::ofstream metadata(path.string() + ".txt");
    metadata << weight << '\t' << trace.phase << '\t' << ggml_type_name(input->type)
             << '\t' << input->ne[0] << '\t' << input->ne[1] << '\t' << data.size() << '\n';
    std::cerr << "traced " << path << " " << input->ne[0] << 'x' << input->ne[1] << '\n';
    return true;
}

int main(int argc, char **argv) {
    if (argc != 4 && argc != 5) fail("usage: task020_score_llama MODEL CASES.tsv SCORES.tsv [TRACE_DIR]");
    ggml_backend_load_all_from_path("/app");
    llama_backend_init();
    auto model_params = llama_model_default_params();
    model_params.n_gpu_layers = -1;
    auto *gpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!gpu) fail("GPU backend is unavailable");
    llama_model_tensor_buft_override buffer_overrides[] = {
        {"token_embd.weight", ggml_backend_dev_buffer_type(gpu)}, {nullptr, nullptr}};
    model_params.tensor_buft_overrides = buffer_overrides;
    llama_model *model = llama_model_load_from_file(argv[1], model_params);
    if (!model) fail("failed to load model");
    const int vocab_size = llama_vocab_n_tokens(llama_model_get_vocab(model));
    auto context_params = llama_context_default_params();
    context_params.n_ctx = 1024;
    context_params.n_batch = 512;
    context_params.n_ubatch = 512;
    context_params.n_seq_max = 1;
    context_params.no_perf = true;
    Trace trace;
    if (argc == 5) {
        trace.directory = argv[4];
        std::filesystem::create_directories(trace.directory);
        context_params.cb_eval = trace_input;
        context_params.cb_eval_user_data = &trace;
    }
    llama_context *ctx = llama_init_from_model(model, context_params);
    if (!ctx) fail("failed to create GPU context");
    llama_batch batch = llama_batch_init(512, 0, 1);
    std::ifstream cases(argv[2]);
    std::ofstream scores(argv[3]);
    if (!cases || !scores) fail("failed to open cases or scores file");
    scores << "id\tprompt_sha256\ttarget_sha256\tprompt_tokens\ttarget_tokens\tnll\tfirst_match\tgreedy_lcp\n";
    scores << std::setprecision(17);
    std::string line;
    int count = 0;
    while (std::getline(cases, line)) {
        auto row = fields(line);
        if (row.size() != 4) fail("invalid cases row");
        auto tokens = read_tokens(row[1]);
        llama_memory_clear(llama_get_memory(ctx), true);
        trace.phase = "prefill";
        decode(ctx, batch, tokens.data(), 384, 0, true);
        double nll = 0.0;
        int lcp = 0;
        bool matching = true;
        for (int i = 0; i < 128; ++i) {
            const float *logits = llama_get_logits_ith(ctx, -1);
            if (!logits) fail("missing logits");
            llama_token greedy = 0;
            nll += negative_log_probability(logits, vocab_size, tokens[384 + i], greedy);
            if (matching && greedy == tokens[384 + i]) ++lcp;
            else matching = false;
            if (i < 127) {
                trace.phase = "decode";
                decode(ctx, batch, &tokens[384 + i], 1, 384 + i, true);
            }
        }
        scores << row[0] << '\t' << row[2] << '\t' << row[3]
               << "\t384\t128\t" << nll << '\t' << (lcp > 0) << '\t' << lcp << '\n';
        scores.flush();
        std::cerr << row[0] << " avg_nll=" << nll / 128 << " lcp=" << lcp << '\n';
        ++count;
    }
    std::cerr << "scored_cases=" << count << '\n';
    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
}
