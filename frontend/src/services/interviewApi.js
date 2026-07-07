const API_BASE_URL = 'https://fantastic-bassoon-97qpqqwjp56vh9vr6-8000.app.github.dev/api/v1';

export const uploadResumeFile = async (file) => {
  const formData = new FormData();
  formData.append("file", file);

  try {
    const response = await fetch(`${API_BASE_URL}/resume/upload`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`Upload failed with status: ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    console.error("Resume uploading failed:", error);
    throw error;
  }
};

export const fetchInterviewResponse = async (userInput, chatHistory, currentPhase) => {
  try {
    const response = await fetch(`${API_BASE_URL}/session/respond`, {
      method: 'POST',
      headers: {
        'Accept': 'application/json',
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        user_input: userInput,
        chat_history: chatHistory.map(msg => ({
          role: msg.role === 'assistant' ? 'assistant' : 'user',
          content: msg.content
        })),
        current_phase: currentPhase
      }),
    });

    if (!response.ok) {
      throw new Error(`HTTP Error: ${response.status}`);
    }

    const data = await response.json();
    return {
      aiMessage: data.response?.[0]?.text || "No response text found.",
      nextPhase: data.current_phase || currentPhase,
      evaluation: data.evaluation || { technical_accuracy: 0.0 }
    };
  } catch (error) {
    console.error("Backend communication failed:", error);
    throw error;
  }
};