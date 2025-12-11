# Installing Ollama on Windows

## Quick Installation Steps

1. **Download Ollama:**
   - Visit: https://ollama.com/download
   - Click "Download for Windows"
   - Save the `OllamaSetup.exe` file

2. **Run the Installer:**
   - Double-click `OllamaSetup.exe`
   - Follow the installation wizard
   - Ollama will be installed and started automatically

3. **Verify Installation:**
   ```powershell
   ollama --version
   ```

4. **Pull a Model (Required for RAG system):**
   ```powershell
   ollama pull llama3.2:3b
   ```
   
   Or for a smaller/faster model:
   ```powershell
   ollama pull phi3:mini
   ```

5. **Verify Model is Available:**
   ```powershell
   ollama list
   ```

## After Installation

Once Ollama is installed and a model is pulled, your RAG system will automatically use it (since `USE_OLLAMA=true` is now the default).

The system will:
- Connect to Ollama at `http://localhost:11434`
- Use the model specified in your `.env` file (default: `llama3.2:3b`)
- Provide 5-10x faster generation compared to local TinyLlama

## Troubleshooting

If Ollama is not found after installation:
- Restart your terminal/PowerShell
- Check if Ollama service is running (it should start automatically)
- Verify installation path is in your PATH environment variable

