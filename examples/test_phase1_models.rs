use llmkit::models::get_model_info;

fn main() {
    // Test some Phase 1 models
    let test_models = vec![
        ("bytedance-seed/seed-1.6-flash", "ByteDance Seed 1.6 Flash"),
        ("openai/gpt-5.2-pro", "OpenAI GPT-5.2 Pro"),
        ("google/gemini-3-pro", "Google Gemini 3 Pro"),
        ("amazon/nova-2-lite-v1", "Amazon Nova 2 Lite"),
        ("anthropic/claude-opus-4-5", "Claude Opus 4.5"),
    ];

    let mut found_count = 0;

    for (model_id, _expected_name) in &test_models {
        if let Some(info) = get_model_info(model_id) {
            println!("✓ Found: {} -> {}", model_id, info.name);
            found_count += 1;
        } else {
            println!("✗ Not found: {}", model_id);
        }
    }

    println!(
        "\nFound {} out of {} test models",
        found_count,
        test_models.len()
    );
}
