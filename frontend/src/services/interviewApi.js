import axios from 'axios';

// GitHub Codespaces dynamic forward URL fallback logic
const getBaseUrl = () => {
  return "https://fantastic-bassoon-97qpqqwjp56vh9vr6-8000.app.github.dev";
};

const api = axios.create({
  baseURL: getBaseUrl(),
  headers: {
    'Content-Type': 'application/json',
  },
});

export const sendInterviewResponse = async (userInput, chatHistory, currentPhase) => {
  try {
    const response = await api.post('/api/v1/session/respond', {
      user_input: userInput,
      chat_history: chatHistory,
      current_phase: currentPhase
    });
    return response.data;
  } catch (error) {
    console.error("Backend communication failed:", error);
    throw error;
  }
};

export const normalizeInterviewResponse = (payload) => {
  return {
    currentPhase: payload.current_phase ?? 'warmup',
    technicalAccuracy: payload.technical_accuracy ?? payload.evaluation?.technical_accuracy ?? 0,
    summary: payload.summary ?? payload.response ?? '',
    messages: Array.isArray(payload.messages) ? payload.messages : [],
    response: payload.response ?? '',
  };
};

export const uploadResume = async (file) => {
  try {
    const formData = new FormData();
    formData.append('file', file);
    const response = await api.post('/api/v1/resume/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  } catch (error) {
    console.error("Resume upload failed:", error);
    throw error;
  }
};
