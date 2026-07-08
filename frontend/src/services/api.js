const API_BASE_URL = 'https://fantastic-bassoon-97qpqqwjp56vh9vr6-8000.app.github.dev/';

export const sendInterviewResponse = async (userInput, chatHistory, currentPhase) => {
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
      throw new Error(`Server status returned: ${response.status}`);
    }

    const data = await response.json();
    return {
      aiMessage: data.response?.[0]?.text || "No response text found.",
      nextPhase: data.current_phase || currentPhase,
      evaluation: data.evaluation || { technical_accuracy: 0.0 }
    };
  } catch (error) {
    console.error("API call error:", error);
    throw error;
  }
};