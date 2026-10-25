# Java 调用 vLLM OpenAI 兼容 API 完整示例（L09 配套）

> 面向 Spring Boot 技术栈的接入骨架。vLLM 的 `/v1/chat/completions` 与 OpenAI API 格式一致，
> 字段细节以 OpenAI API 文档与 vLLM 官方文档为准，版本可能更新。

## 0. 前置：vLLM 服务已就绪

```bash
# 启动自托管模型（含你微调合并后的模型；典型示例，参数见 code/../04 的 vLLM 速查）
vllm serve /path/to/models/qwen25-7b-mytask-merged \
    --served-model-name java-assistant \
    --host 0.0.0.0 --port 8000 \
    --max-model-len 8192

# 冒烟验证
curl http://llm-server:8000/v1/models
```

## 1. 配置（application.properties）

```properties
# LLM 服务地址与模型名（生产放配置中心，便于切换/灰度）
llm.base-url=http://llm-server:8000/v1
llm.model=java-assistant
llm.api-key=EMPTY
# 采样参数按业务定死默认值（代码助手类任务低温，见阶段4 L03 场景表）
llm.temperature=0.3
llm.max-tokens=1024
# LLM 生成慢：连接超时短、响应超时给足
llm.connect-timeout-seconds=5
llm.read-timeout-seconds=300
```

## 2. 配置类与核心服务

```java
// LlmProperties.java —— 配置绑定
@ConfigurationProperties(prefix = "llm")
public record LlmProperties(
        String baseUrl, String model, String apiKey,
        double temperature, int maxTokens,
        int connectTimeoutSeconds, int readTimeoutSeconds) {}
```

```java
// LlmService.java —— 核心：HttpClient 封装 /v1/chat/completions
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.stereotype.Service;
import java.net.URI;
import java.net.http.*;
import java.time.Duration;
import java.util.List;
import java.util.Map;

@Service
public class LlmService {

    /** 请求体对应的 DTO 结构（只声明用到的字段） */
    public record ChatMessage(String role, String content) {
        public static ChatMessage system(String c) { return new ChatMessage("system", c); }
        public static ChatMessage user(String c)   { return new ChatMessage("user", c); }
    }

    @JsonIgnoreProperties(ignore_unknown = true)
    static class ChatCompletionResponse {
        public List<Choice> choices;
        public Usage usage;                       // token 用量统计（计费/监控可用）
    }
    @JsonIgnoreProperties(ignore_unknown = true)
    static class Choice { public ChatMessage message; }
    @JsonIgnoreProperties(ignore_unknown = true)
    static class Usage { public int prompt_tokens, completion_tokens, total_tokens; }

    private final HttpClient http;
    private final ObjectMapper mapper = new ObjectMapper();
    private final LlmProperties props;

    public LlmService(LlmProperties props) {
        this.props = props;
        this.http = HttpClient.newBuilder()
                .connectTimeout(Duration.ofSeconds(props.connectTimeoutSeconds()))
                .build();
    }

    /** 非流式对话：等全部生成完一次性返回（简单场景/内部任务） */
    public String chat(List<ChatMessage> messages) {
        try {
            var body = mapper.createObjectNode();
            body.put("model", props.model());
            body.set("messages", mapper.valueToTree(messages));
            body.put("temperature", props.temperature());
            body.put("max_tokens", props.maxTokens());
            body.put("stream", false);

            HttpRequest request = HttpRequest.newBuilder()
                    .uri(URI.create(props.baseUrl() + "/chat/completions"))
                    .header("Content-Type", "application/json")
                    .header("Authorization", "Bearer " + props.apiKey()) // vLLM 默认不校验，传占位即可
                    .timeout(Duration.ofSeconds(props.readTimeoutSeconds()))
                    .POST(HttpRequest.BodyPublishers.ofString(body.toString()))
                    .build();

            HttpResponse<String> resp = http.send(request, HttpResponse.BodyHandlers.ofString());
            if (resp.statusCode() != 200) {
                throw new IllegalStateException("LLM HTTP " + resp.statusCode() + ": " + resp.body());
            }
            return mapper.readValue(resp.body(), ChatCompletionResponse.class)
                         .choices.get(0).message.content();
        } catch (Exception e) {
            // 降级策略放这里（L09 checklist：失败路径必须有设计）
            throw new IllegalStateException("调用 LLM 服务失败", e);
        }
    }
}
```

## 3. REST Controller（含 SSE 流式）

```java
// ChatController.java —— 对外接口
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.net.URI;
import java.net.http.*;
import java.time.Duration;
import java.util.List;

@RestController
@RequestMapping("/api/chat")
public class ChatController {

    private final LlmService llmService;
    private final LlmProperties props;
    private final HttpClient http = HttpClient.newHttpClient();
    private final com.fasterxml.jackson.databind.ObjectMapper mapper =
            new com.fasterxml.jackson.databind.ObjectMapper();

    public ChatController(LlmService llmService, LlmProperties props) {
        this.llmService = llmService;
        this.props = props;
    }

    /** 非流式：POST /api/chat  body: {"question": "..."} */
    public record ChatRequest(String question) {}

    @PostMapping
    public Map<String, String> chat(@RequestBody ChatRequest req) {
        String answer = llmService.chat(List.of(
                LlmService.ChatMessage.system("你是团队的 Java 编程助手，遵循团队编码规范。"),
                LlmService.ChatMessage.user(req.question())));
        return Map.of("answer", answer);
    }

    /** 流式：POST /api/chat/stream —— SSE 转发 vLLM 的 stream:true 响应（打字机效果） */
    @PostMapping(value = "/stream", produces = MediaType.TEXT_EVENT_STREAM_VALUE)
    public SseEmitter chatStream(@RequestBody ChatRequest req) {
        SseEmitter emitter = new SseEmitter(0L);   // 不设超时（生成时长不定）
        new Thread(() -> {
            try {
                var body = mapper.createObjectNode();
                body.put("model", props.model());
                body.set("messages", mapper.valueToTree(List.of(
                        LlmService.ChatMessage.user(req.question()))));
                body.put("stream", true);

                HttpRequest request = HttpRequest.newBuilder()
                        .uri(URI.create(props.baseUrl() + "/chat/completions"))
                        .header("Content-Type", "application/json")
                        .POST(HttpRequest.BodyPublishers.ofString(body.toString()))
                        .build();

                // 逐行读 SSE：每行形如 data: {...}，最后以 data: [DONE] 结束
                HttpResponse<java.io.InputStream> resp = http.send(
                        request, HttpResponse.BodyHandlers.ofInputStream());
                try (var reader = new BufferedReader(
                        new InputStreamReader(resp.body(), java.nio.charset.StandardCharsets.UTF_8))) {
                    String line;
                    while ((line = reader.readLine()) != null) {
                        if (!line.startsWith("data: ")) continue;
                        String payload = line.substring(6);
                        if ("[DONE]".equals(payload)) break;
                        var delta = mapper.readTree(payload)
                                .path("choices").path(0)
                                .path("delta").path("content").asText("");
                        if (!delta.isEmpty()) {
                            emitter.send(SseEmitter.event().data(delta));
                        }
                    }
                }
                emitter.complete();
            } catch (Exception e) {
                emitter.completeWithError(e);
            }
        }).start();
        return emitter;
    }
}
```

## 4. 验证

```bash
# 非流式
curl -X POST http://localhost:8080/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "写一个线程安全的单例"}'

# 流式（SSE，逐段返回）
curl -N -X POST http://localhost:8080/api/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "解释 JVM 双亲委派模型"}'
```

## 5. 工程要点清单（LLM 接入的特有项）

| 事项 | 说明 |
|---|---|
| 超时 | 生成时长与 max_tokens 正比：非流式给宽（分钟级）；流式首字节快，但连接保持久 |
| 幂等性 | LLM 生成不幂等：重试会得到不同结果，重试策略只放网络层（连接失败），不放生成层 |
| 限流 | GPU 是瓶颈资源：上游限流（如每用户 QPS）保护推理服务，压测后再定阈值 |
| 监控 | 记录 usage（prompt/completion tokens）做成本监控；TTFT 与 P95 延迟接入告警 |
| 降级 | LLM 服务不可用时返回兜底话术/排队提示，别让上游级联超时 |
| 环境切换 | base-url 配置化：本地 Ollama/云 vLLM/外部 API 一键切换（策略模式抽象 LlmClient） |
