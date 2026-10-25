# L04 配套：Java HttpClient 对接 Agent 服务

> 目标读者：Java Web 老兵（就是你）。零第三方依赖——只要 JDK 11+（建议 17+，用到了文本块），用原生 `java.net.http.HttpClient` 完成对 `L04_fastapi_server.py` 的一次性调用与 SSE 流式消费。
> 服务端先跑起来：`python L04_fastapi_server.py`（默认 `http://127.0.0.1:8000`）。

## 一、完整客户端代码

保存为 `AgentClient.java`，编译运行：

```bash
javac AgentClient.java
java AgentClient http://127.0.0.1:8000     # 参数：服务基地址
```

```java
// 文件：AgentClient.java
// 环境要求：JDK 17+（文本块）；无任何第三方依赖
// 以 JDK 官方文档为准（java.net.http.HttpClient 自 JDK 11 起可用）

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.stream.Stream;

public class AgentClient {

    public static void main(String[] args) throws Exception {
        String base = args.length > 0 ? args[0] : "http://127.0.0.1:8000";

        // HttpClient 实例可全局复用（线程安全），这里演示用，只建一次
        HttpClient client = HttpClient.newBuilder()
                .connectTimeout(Duration.ofSeconds(5))
                .build();

        System.out.println("== 1. 一次性调用 /chat ==");
        chatOnce(client, base, "用一句话介绍你自己");

        System.out.println("\n== 2. 流式消费 /chat/stream（SSE）==");
        chatStream(client, base, "流式讲个两句话的笑话", "session-java-1");
    }

    /** 一次性调用：发 JSON、收 JSON——你每天在写的普通 HTTP 调用 */
    static void chatOnce(HttpClient client, String base, String message) throws Exception {
        String body = """
                {"message": "%s", "session_id": "session-java-1"}
                """.formatted(escapeJson(message));

        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(base + "/chat"))
                .timeout(Duration.ofSeconds(30))          // 模型回答可能很慢，超时给足
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(body))
                .build();

        HttpResponse<String> response =
                client.send(request, HttpResponse.BodyHandlers.ofString());

        if (response.statusCode() == 200) {
            System.out.println("reply: " + response.body());
        } else if (response.statusCode() == 422) {
            // 参数校验失败：FastAPI/Pydantic 返回结构化错误，按需解析处理
            System.out.println("校验失败(422): " + response.body());
        } else {
            System.out.println("异常状态码 " + response.statusCode() + ": " + response.body());
        }
    }

    /**
     * 流式消费：SSE 本质是"响应体逐行到达"。
     * ofLines() 返回 Stream<String>，服务端每推一行，这里就收到一行——
     * 这就是打字机效果的客户端半边。注意：浏览器 EventSource 只支持 GET，
     * 服务端互调用 POST + ofLines 完全没问题（主流大模型 API 同款组合）。
     */
    static void chatStream(HttpClient client, String base, String message,
                           String sessionId) throws Exception {
        String body = """
                {"message": "%s", "session_id": "%s"}
                """.formatted(escapeJson(message), escapeJson(sessionId));

        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(base + "/chat/stream"))
                .timeout(Duration.ofSeconds(60))          // 注意：这是整个响应的超时上限
                .header("Content-Type", "application/json")
                .header("Accept", "text/event-stream")    // 声明期望 SSE
                .POST(HttpRequest.BodyPublishers.ofString(body))
                .build();

        // 关键：BodyHandlers.ofLines() 把响应体按行流式送达，而不是攒完再给
        HttpResponse<Stream<String>> response =
                client.send(request, HttpResponse.BodyHandlers.ofLines());

        if (response.statusCode() != 200) {
            System.out.println("流式请求失败，状态码 " + response.statusCode());
            return;
        }

        StringBuilder full = new StringBuilder();
        try (Stream<String> lines = response.body()) {    // Stream 用完要关（释放连接）
            lines.forEach(line -> {
                // SSE 协议：事件行以 "data: " 开头，行间空行是分隔符（细节以规范为准）
                if (!line.startsWith("data: ")) {
                    return;                               // 跳过空行/注释行
                }
                String payload = line.substring("data: ".length()).trim();
                // 这里拿到的是 {"delta":"...","done":false}；无 JSON 库时做最轻量解析
                String delta = extractField(payload, "delta");
                String compact = payload.replaceAll("\\s", "");   // 去空白便于模式匹配
                boolean done = compact.matches(".*\"done\":true.*");
                if (!done && !delta.isEmpty()) {
                    System.out.print(delta);              // 打字机效果：来一块打一块
                    System.out.flush();
                    full.append(delta);
                } else if (done) {
                    System.out.println("\n[流式完成，共 " + full.length() + " 字符]");
                }
            });
        }
    }

    /** 最小 JSON 字符串转义（生产建议上 Jackson/Gson，这里保持零依赖） */
    static String escapeJson(String s) {
        return s.replace("\\", "\\\\")
                .replace("\"", "\\\"")
                .replace("\n", "\\n")
                .replace("\r", "\\r")
                .replace("\t", "\\t");
    }

    /** 零依赖时抽取简单字段的演示实现；生产请换真正的 JSON 解析器 */
    static String extractField(String json, String field) {
        String needle = "\"" + field + "\":";
        int i = json.indexOf(needle);
        if (i < 0) {
            return "";
        }
        int start = json.indexOf('"', i + needle.length()) + 1;
        int end = json.indexOf('"', start);
        return end > start ? json.substring(start, end) : "";
    }
}
```

## 二、要点速记（都是你熟悉的概念换了名字）

| Java 侧 | 说明 |
|---|---|
| `HttpClient` 线程安全可复用 | 别每次请求 new 一个——连接池白白浪费 |
| `ofString()` vs `ofLines()` | 一次性 vs 流式；对应服务端普通 JSON 与 SSE |
| `timeout()` 语义 | 连接/请求级超时；**流式场景要按"整个回答时长"给** |
| 422 与 200 之外的分支 | Pydantic 校验错误是结构化 JSON，可以像处理 `MethodArgumentNotValidException` 一样转成友好提示 |
| SSE 的 `data: ` 前缀 | 协议固定格式；空行是事件分隔，解析时跳过 |

## 三、自测核对

跑通后你应该看到：

1. 第 1 段：一次性打印完整 `reply`（JSON 字符串，含 `latency_ms`）
2. 第 2 段：**逐块**打印的打字机效果，最后一行 `[流式完成，共 N 字符]`
3. 停掉 Python 服务再跑客户端：应进入异常路径（`ConnectException`）——这是练习 2 的起点：给 `chatOnce` 加"仅幂等读操作"的重试

## 四、常见坑

- **看不到逐块效果**：客户端或中间代理做了缓冲；curl 侧记得 `-N`
- **中文乱码**：确认终端与文件编码都是 UTF-8；`HttpClient` 收发的字节流编码跟随响应头，服务端已显式 UTF-8
- **JDK 版本**：文本块（`"""`）需要 15+，`formatted()` 需要 15+；JDK 11 用户把两处改为字符串拼接即可，其余 API 不变
- **生产化**：上 JSON 库替换手写解析、加熔断与指标埋点——这些你在 Java 侧的老手艺不用我教
