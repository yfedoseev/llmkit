/**
 * LLMKit Audio Synthesis Example (TypeScript)
 *
 * This example demonstrates how to use the LLMKit Node.js bindings to synthesize
 * speech from text using ElevenLabs provider.
 *
 * Requirements:
 * - llmkit npm package
 * - ELEVENLABS_API_KEY environment variable
 * - Speaker voice ID (available from ElevenLabs API)
 *
 * Usage:
 *     npx ts-node typescript_audio_synthesis.ts "Your text here" [output_file]
 */

import fs from 'fs';
import path from 'path';
import { LLMKitClient, SynthesisRequest, SynthesizeOptions, VoiceSettings, LatencyMode } from 'llmkit';

/**
 * Synthesize speech from text using ElevenLabs
 */
async function synthesizeSpeech(
  client: LLMKitClient,
  text: string,
  outputFile: string
): Promise<void> {
  console.log('\n' + '='.repeat(70));
  console.log('TEXT-TO-SPEECH SYNTHESIS');
  console.log('='.repeat(70));

  console.log(`\nInput text: ${text}`);

  try {
    // Create synthesis request
    const request = new SynthesisRequest(text);

    // Optionally set voice ID (example uses Rachel)
    // Voice IDs available: 21m00Tcm4TlvDq8ikWAM (Rachel), pNInY14gQrG92XwBIHVr, etc.
    request.with_voice('21m00Tcm4TlvDq8ikWAM');

    // Create synthesis options
    const options = new SynthesizeOptions();
    options.with_latency_mode(LatencyMode.Balanced);
    options.with_output_format('mp3_44100_64');

    // Optionally customize voice settings
    const voiceSettings = new VoiceSettings(0.5, 0.75);
    options.with_voice_settings(voiceSettings);

    // Synthesize speech
    console.log('\nSynthesizing speech...');
    const response = await client.synthesize_speech(request);

    // Display results
    console.log('\nSynthesis complete!');
    console.log(`  Format: ${response.format}`);
    console.log(`  Size: ${response.size} bytes`);
    if (response.duration) {
      console.log(`  Duration: ${response.duration.toFixed(2)}s`);
    }

    // Save to file
    console.log(`\nSaving audio to: ${outputFile}`);
    const buffer = Buffer.from(response.audio_bytes);
    fs.writeFileSync(outputFile, buffer);

    const stats = fs.statSync(outputFile);
    console.log(`✓ Audio file saved successfully!`);
    console.log(`  File size: ${(stats.size / 1024).toFixed(1)} KB`);
  } catch (error) {
    console.error(`Error during synthesis: ${error}`);
    throw error;
  }
}

/**
 * Main example function
 */
async function main(): Promise<void> {
  // Parse arguments
  if (process.argv.length < 3) {
    console.log("Usage: npx ts-node typescript_audio_synthesis.ts 'Your text here' [output_file]");
    console.log('\nExamples:');
    console.log("  npx ts-node typescript_audio_synthesis.ts 'Hello, world!'");
    console.log("  npx ts-node typescript_audio_synthesis.ts 'Tell me a joke' output.mp3");
    process.exit(1);
  }

  const text = process.argv[2];
  const outputFile = process.argv[3] ?? 'synthesized.mp3';

  // Check if output file already exists
  if (fs.existsSync(outputFile)) {
    const prompt = `File ${outputFile} already exists. Overwrite? [y/N]: `;
    console.log(prompt);
    // Note: In a real CLI app, you'd use a library like readline or prompts
    // For this example, we'll just overwrite
    console.log('Overwriting existing file...');
  }

  // Initialize LLMKit client
  console.log('Initializing LLMKit client...');
  let client: LLMKitClient;
  try {
    client = LLMKitClient.fromEnv();
  } catch (error) {
    console.error(`Error initializing client: ${error}`);
    console.log('\nMake sure you have set the required environment variable:');
    console.log('  - ELEVENLABS_API_KEY');
    process.exit(1);
  }

  // Synthesize speech
  await synthesizeSpeech(client, text, outputFile);

  console.log('\n' + '='.repeat(70));
  console.log('Synthesis example complete!');
  console.log('='.repeat(70));
}

// Run the example
main().catch((error) => {
  console.error('Fatal error:', error);
  process.exit(1);
});
