# IraAI Inference Server

The dedicated local inference server for IraAI.

This server is responsible for loading and running IraAI's
machine-learning models.

The Render backend does not load models.

---

## Architecture

```text
IraAI Android App
        |
        v
Render Backend
        |
        v
IraAI Inference Server
        |
        +---- Hugging Face Model Repository
        |
        v
Local Model Inference
        |
        v
Render Backend
        |
        v
IraAI Android App