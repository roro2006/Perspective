"""Voice bridge to Curio-for-screens (clicky-py): push-to-talk, Whisper
transcription, Kokoro TTS and the OpenAI-compatible client for the lab VLM.

TODO(owner C): import from clicky.clients (whisper_transcription_client,
kokoro_tts_client, openai_llm_client) and expose:
    listen() -> str            transcript of one push-to-talk utterance
    say(text: str) -> None     spoken confirmation ("Got it, the red one on your left")
"""
