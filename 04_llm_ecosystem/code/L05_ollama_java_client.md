# L05 配套：Java HttpClient 调用 Ollama API 完整示例

> 面向 Java Web 开发者的零依赖（JDK 11+ 自带 HttpClient）本地 LLM 集成示例。
> 请求/响应字段以 Ollama 官方 API 文档为准（GitHub 仓库 ollama/ollama 的 `docs/api.md`），版本可能更新。

## 0. 前置准备

```bash
# 1. 安装并启动 Ollama（Linux，以官网安装页为准）
curl -fsSL https://ollama.com/install.sh | sh

# 2. 拉一个小模型（CPU 也可流畅运行）
ollama pull qwen2.5:0.5b

# 3. 确认服务在 11434 端口
curl http://localhost:11434/api/tags
```

## 1. 最小可运行示例：单文件直跑（无任何第三方依赖）

保存为 `OllamaDemo.java`，`java OllamaDemo.java` 直接运行（JDK 11+ 单文件运行）：

```java
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;

public class OllamaDemo {

    public static void main(String[] args) throws Exception {
        // Ollama 默认监听 localhost:11434；局域网访问需在服务端设置 OLLAMA_HOST=0.0.0.0
        String url = "http://localhost:11434/api/chat";

        // 请求体：多轮消息格式（与 OpenAI 的 messages 结构同构，迁移成本低）
        // stream=false：等待完整响应，一次返回（对流式输出见第 3 节）
        String requestBody = """
                {
                  "model": "qwen2.5:0.5b",
                  "messages": [
                    {"role": "system", "content": "你是一个严谨的 Java 编程助手，回答不超过三句话。"},
                    {"role": "user", "content": "用一句话解释什么是 KV Cache"}
                  ],
                  "stream": false,
                  "options": { "temperature": 0.7, "top_p": 0.9 }
                }
                """;

        HttpClient client = HttpClient.newBuilder()
                .connectTimeout(Duration.ofSeconds(10))
                .build();

        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(url))
                .header("Content-Type", "application/json")
                .timeout(Duration.ofMinutes(5))   // LLM 生成慢，超时要给足
                .POST(HttpRequest.BodyPublishers.ofString(requestBody))
                .build();

        HttpResponse<String> response = client.send(
                request, HttpResponse.BodyHandlers.ofString());

        System.out.println("HTTP " + response.statusCode());
        System.out.println(response.body());
        // 响应结构（非流式）：{"message": {"role": "assistant", "content": "..."}, "done": true, ...}
        // 生产代码建议用 Jackson/Jackson-databind 解析为 DTO，而非字符串截取
    }
}
```

## 2. 推荐工程形态：封装客户端类 + Jackson 解析

实际项目建议用 Jackson（Maven 坐标示例，版本以 Maven Central 最新稳定版为准）：

```xml
<dependency>
    <groupId>com.fasterxml.jackson.core</groupId>
    <artifactId>jackson-databind</artifactId>
    <version>2.17.1</version>
</dependency>
```

```java
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.net.URI;
import java.net.http.*;
import java.time.Duration;
import java.util.List;

/** Ollama 最小客户端封装：面向应用的 chat 接口 */
public class OllamaClient {

    @JsonIgnoreProperties(ignore_unknown = true)
    public static class ChatMessage {
        public String role;
        public String content;
        public ChatMessage() {}                       // Jackson 反序列化需要
        public ChatMessage(String role, String content) {
            this.role = role;
            this.content = content;
        }
    }

    @JsonIgnoreProperties(ignore_unknown = true)
    static class ChatResponse {                       // 只声明关心的字段，其余忽略
        public ChatMessage message;
        public boolean done;
    }

    private final HttpClient http = HttpClient.newBuilder()
            .connectTimeout(Duration.ofSeconds(10)).build();
    private final ObjectMapper mapper = new ObjectMapper();
    private final String baseUrl;                     // 例如 http://192.168.1.10:11434

    public OllamaClient(String baseUrl) { this.baseUrl = baseUrl; }

    /** 多轮对话：传入完整历史，返回助手新回复 */
    public String chat(String model, List<ChatMessage> messages) throws Exception {
        var body = mapper.createObjectNode();
        body.put("model", model);
        body.set("messages", mapper.valueToTree(messages));
        body.put("stream", false);

        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(baseUrl + "/api/chat"))
                .header("Content-Type", "application/json")
                .timeout(Duration.ofMinutes(5))       // 生成耗时与 max token 数正相关
                .POST(HttpRequest.BodyPublishers.ofString(body.toString()))
                .build();

        HttpResponse<String> resp = http.send(request, HttpResponse.BodyHandlers.ofString());
        if (resp.statusCode() != 200) {
            throw new IllegalStateException("Ollama HTTP " + resp.statusCode() + ": " + resp.body());
        }
        return mapper.readValue(resp.body(), ChatResponse.class).message.content;
    }
}
```

调用方式：

```java
var client = new OllamaClient("http://localhost:11434");
String answer = client.chat("qwen2.5:0.5b", List.of(
        new OllamaClient.ChatMessage("user", "写一个线程安全的 Java 单例")));
System.out.println(answer);
```

## 3. 流式输出（NDJSON）：一行一个 JSON 增量

Ollama 流式模式（`"stream": true`，默认即流式）返回 NDJSON——每行一个 JSON，增量包含在
`message.content` 中。用 `BodyHandlers.ofLines()` 逐行消费，体感就是「打字机效果」：

```java
HttpRequest request = HttpRequest.newBuilder()
        .uri(URI.create("http://localhost:11434/api/chat"))
        .header("Content-Type", "application/json")
        .POST(HttpRequest.BodyPublishers.ofString("""
            {"model": "qwen2.5:0.5b",
             "messages": [{"role": "user", "content": "用 200 字介绍 JVM 垃圾回收"}],
             "stream": true}"""))
        .build();

// ofLines() 返回 Stream<String>，每行是一个独立的 JSON 对象（NDJSON 协议）
http.sendAsync(request, HttpResponse.BodyHandlers.ofLines())
    .thenApply(HttpResponse::body)
    .thenAccept(lines -> lines.forEach(line -> {
        try {
            var node = mapper.readTree(line);          // 逐行解析
            System.out.print(node.path("message").path("content").asText());
        } catch (Exception ignored) {}
    }))
    .join();
```

若在 Spring Boot 中暴露给前端，可把该逐行消费改为 `SseEmitter` / `Flux<String>` 转发，
与转发其他 LLM 的流式接口写法完全一致。

## 4. 常见问题速查

| 现象 | 排查 |
|---|---|
| Connection refused | Ollama 未启动；确认 `systemctl status ollama` 或进程存在 |
| 远程机器连不上 | 服务端需 `OLLAMA_HOST=0.0.0.0` 后重启；检查防火墙 11434 端口 |
| 超时（HttpTimeoutException） | 生成时间长属正常：调大 request timeout，或改用流式尽早拿到首字节 |
| 响应中文乱码 | 确认按 UTF-8 处理 body（ofString/ofLines 默认按响应头解码，一般无问题） |
| 想换自己微调的模型 | 用 Modelfile + `ollama create` 注册后，`model` 字段填你起的名字（阶段 5 L09） |

## 5. 与 vLLM 的对比提醒

Ollama 自有 API 与 OpenAI 格式不同（`/api/chat` vs `/v1/chat/completions`）。若你的服务
未来要在 Ollama（本地）与 vLLM（生产）之间切换，可以：

- 直接统一走 OpenAI 兼容格式（vLLM 原生 + Ollama 也有兼容层的路线，以各自官方文档为准），或
- 在 Java 侧抽象一个 `LlmClient` 接口，两个实现类分别对接（经典策略模式，你最熟悉）。
