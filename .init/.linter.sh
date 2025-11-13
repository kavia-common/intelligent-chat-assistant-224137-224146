#!/bin/bash
cd /home/kavia/workspace/code-generation/intelligent-chat-assistant-224137-224146/ai_chat_box_frontend
npm run build
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
   exit 1
fi

