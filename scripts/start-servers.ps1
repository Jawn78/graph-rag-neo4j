# Load configuration
$config = Get-Content -Path "../config/server_config.json" | ConvertFrom-Json

# Start Chat Server
$chatArgs = @(
    "-m", "llama_cpp.server",
    "--model", $config.chat_server.model_path,
    "--host", $config.chat_server.host,
    "--port", $config.chat_server.port,
    "--n_gpu_layers", $config.chat_server.n_gpu_layers,
    "--n_threads", $config.chat_server.n_threads,
    "--chat_format", $config.chat_server.chat_format,
    "--n_ctx", $config.chat_server.n_ctx,
    "--n_batch", $config.chat_server.n_batch
)
if ($config.chat_server.verbose) { $chatArgs += "--verbose", "true" }

Start-Process -NoNewWindow -FilePath "python" -ArgumentList $chatArgs

# Start Embedding Server
$embedArgs = @(
    "-m", "llama_cpp.server",
    "--model", $config.embedding_server.model_path,
    "--host", $config.embedding_server.host,
    "--port", $config.embedding_server.port,
    "--model_alias", $config.embedding_server.model_alias,
    "--n_gpu_layers", $config.embedding_server.n_gpu_layers,
    "--n_threads", $config.embedding_server.n_threads
)
if ($config.embedding_server.embedding) { $embedArgs += "--embedding", "true" }
if ($config.embedding_server.verbose) { $embedArgs += "--verbose", "true" }

Start-Process -NoNewWindow -FilePath "python" -ArgumentList $embedArgs

Write-Host "Servers started. Chat server on port $($config.chat_server.port), Embedding server on port $($config.embedding_server.port)"