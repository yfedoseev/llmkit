/**
 * LLMKit Audio Transcription Example (TypeScript)
 *
 * This example demonstrates how to use the LLMKit Node.js bindings to transcribe
 * audio files using various providers (Deepgram, AssemblyAI).
 *
 * Requirements:
 * - llmkit npm package
 * - DEEPGRAM_API_KEY or ASSEMBLYAI_API_KEY environment variable
 *
 * Usage:
 *     npx ts-node typescript_audio_transcription.ts <audio_file>
 */

import fs from 'fs';
import path from 'path';
import { LLMKitClient, TranscriptionRequest } from 'modelsuite';

/**
 * Transcribe audio file using Deepgram provider
 */
async function transcribeWithDeepgram(
  client: LLMKitClient,
  audioBytes: Buffer
): Promise<void> {
  console.log('\n' + '='.repeat(70));
  console.log('TRANSCRIPTION WITH DEEPGRAM');
  console.log('='.repeat(70));

  try {
    // Create transcription request
    const request = new TranscriptionRequest(audioBytes);
    request.with_model('nova-3');

    // Transcribe audio
    console.log('Transcribing audio...');
    const response = await client.transcribe_audio(request);

    // Display results
    console.log('\nTranscript:');
    console.log(`  ${response.transcript}`);
    console.log(`\nConfidence: ${(response.confidence ?? 0) * 100}%`);
    if (response.duration) {
      console.log(`Duration: ${response.duration.toFixed(2)}s`);
    }
    console.log(`Word count: ${response.word_count}`);

    // Display word-level details if available
    if (response.words && response.words.length > 0) {
      console.log('\nWord-level details (first 5 words):');
      for (const word of response.words.slice(0, 5)) {
        console.log(
          `  '${word.word}': ${word.start.toFixed(2)}s-${word.end.toFixed(2)}s ` +
            `(confidence: ${(word.confidence * 100).toFixed(1)}%)`
        );
      }
    }
  } catch (error) {
    console.error(`Error during transcription: ${error}`);
    throw error;
  }
}

/**
 * Main example function
 */
async function main(): Promise<void> {
  // Check arguments
  if (process.argv.length < 3) {
    console.log('Usage: npx ts-node typescript_audio_transcription.ts <audio_file>');
    console.log('\nExample:');
    console.log('  npx ts-node typescript_audio_transcription.ts speech.wav');
    process.exit(1);
  }

  const audioPath = path.resolve(process.argv[2]);

  // Check if file exists
  if (!fs.existsSync(audioPath)) {
    console.error(`Error: Audio file not found: ${audioPath}`);
    process.exit(1);
  }

  // Read audio file
  console.log(`Reading audio file: ${audioPath}`);
  const audioBytes = fs.readFileSync(audioPath);
  const fileSizeKB = (audioBytes.length / 1024).toFixed(1);
  console.log(`Audio file size: ${fileSizeKB} KB`);

  // Check file size (warn if very large)
  if (audioBytes.length > 50 * 1024 * 1024) {
    console.warn(`Warning: Large audio file (${(audioBytes.length / 1024 / 1024).toFixed(1)} MB)`);
    console.warn('         Transcription may take longer');
  }

  // Initialize LLMKit client
  console.log('\nInitializing LLMKit client...');
  let client: LLMKitClient;
  try {
    client = LLMKitClient.fromEnv();
  } catch (error) {
    console.error(`Error initializing client: ${error}`);
    console.log('\nMake sure you have set the required environment variables:');
    console.log('  - DEEPGRAM_API_KEY (for Deepgram provider)');
    console.log('  - ASSEMBLYAI_API_KEY (for AssemblyAI provider)');
    process.exit(1);
  }

  // Transcribe with Deepgram
  await transcribeWithDeepgram(client, audioBytes);

  console.log('\n' + '='.repeat(70));
  console.log('Transcription complete!');
  console.log('='.repeat(70));
}

// Run the example
main().catch((error) => {
  console.error('Fatal error:', error);
  process.exit(1);
});
