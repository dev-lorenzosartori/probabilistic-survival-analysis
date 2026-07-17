# ChatGPT GitHub App Server

Um servidor Node.js que integra ChatGPT/OpenAI com seu repositório GitHub através de um GitHub App.

## 🚀 Funcionalidades

- **Code Reviews em PRs**: Análise automática de código em pull requests
- **Sugestões em Issues**: Recomendações inteligentes para novas issues
- **Webhooks do GitHub**: Recebe eventos em tempo real

## 📋 Pré-requisitos

- Node.js 18+
- npm ou yarn
- Credenciais do GitHub App (que você já criou)
- Chave privada do GitHub App (arquivo .pem)
- API Key do OpenAI

## 🔧 Instalação

### 1. Instale as dependências

```bash
cd server
npm install
```

### 2. Configure as variáveis de ambiente

Copie o arquivo `.env.example` para `.env`:

```bash
cp .env.example .env
```

Edite `.env` e preencha com suas credenciais:

```env
GITHUB_APP_ID=seu-app-id
GITHUB_CLIENT_ID=seu-client-id
GITHUB_CLIENT_SECRET=seu-client-secret
GITHUB_INSTALLATION_ID=147113628
GITHUB_WEBHOOK_SECRET=seu-webhook-secret
OPENAI_API_KEY=sua-chave-openai
PORT=3000
```

### 3. Adicione a chave privada do GitHub App

Coloque o arquivo `private-key.pem` (que você baixou ao criar o app) na pasta `server/`:

```bash
# Copie o arquivo private-key.pem para a pasta server
cp ~/Downloads/private-key.pem ./server/
```

### 4. Instale o ngrok (para testar localmente)

```bash
# No macOS
brew install ngrok

# Ou faça download em https://ngrok.com/download
```

## ▶️ Executar o servidor

### Desenvolvimento (com auto-reload)

```bash
npm run dev
```

### Produção

```bash
npm start
```

O servidor rodará em `http://localhost:3000`

## 🌐 Expor localmente com ngrok

Para testar webhooks do GitHub localmente:

```bash
# Em outro terminal
ngrok http 3000
```

Isso fornecerá uma URL pública como: `https://abc123.ngrok.io`

## 📡 Configurar Webhook no GitHub App

1. Vá para https://github.com/settings/apps/seu-app-name
2. Em **Webhook**, atualize:
   - **Webhook URL**: `https://seu-dominio.com/webhook` (ou sua URL do ngrok)
   - **Webhook Secret**: Cole o secret que você tem em `.env`

3. Salve as mudanças

## 📊 Testando

### Verificar saúde do servidor

```bash
curl http://localhost:3000/health
```

### Simular webhook (opcional)

```bash
curl -X POST http://localhost:3000/webhook \
  -H "Content-Type: application/json" \
  -H "X-GitHub-Event: push" \
  -d '{"repository": {"name": "test"}}'
```

## 🚀 Deploy na produção

### Opção 1: Render (recomendado, fácil)

1. Faça push do código para GitHub
2. Vá para https://render.com
3. Clique em "New +" → "Web Service"
4. Conecte seu repositório
5. Configure:
   - **Build Command**: `npm install`
   - **Start Command**: `npm start`
   - Adicione variáveis de ambiente em **Environment**

### Opção 2: Railway

Similar ao Render, suporte nativo para Node.js

### Opção 3: Heroku (pago)

```bash
heroku create seu-app-name
git push heroku main
heroku config:set GITHUB_APP_ID=seu-app-id
# ... etc para todas as variáveis
```

## 📝 Estrutura do código

```
server/
├── index.js                 # Servidor principal
├── package.json            # Dependências
├── .env.example            # Variáveis de ambiente (template)
├── private-key.pem         # Chave privada do GitHub App
└── README.md              # Este arquivo
```

## 🐛 Troubleshooting

### "Missing environment variable"
- Certifique-se de que o arquivo `.env` existe
- Verifique se todas as variáveis estão preenchidas

### "Cannot find module"
- Execute `npm install` novamente
- Delete `node_modules` e `.npm-cache`, execute `npm install`

### Webhook não está sendo acionado
- Verifique se a URL do webhook está correta no GitHub
- Confirm que o firewall/router permite conexões
- Verifique os logs do servidor

### "Invalid signature"
- Certifique-se que `GITHUB_WEBHOOK_SECRET` está correto
- Deve ser igual ao que você configurou no GitHub App

## 📚 Referências

- [GitHub Apps Documentation](https://docs.github.com/en/apps)
- [Octokit.js](https://github.com/octokit/octokit.js)
- [OpenAI API](https://platform.openai.com/docs)
- [Express.js](https://expressjs.com/)

## 📄 Licença

MIT
