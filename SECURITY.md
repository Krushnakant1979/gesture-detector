# Security Policy

## Data Privacy & Zero Cloud Transmission Guarantee

The **ASL Alphabet Fingerspelling Studio** is designed from the ground up with a **strict offline-first, local-processing privacy architecture**:

1. **100% Local Inference**:
   - All live webcam frames, 3D hand landmark extractions, geometric normalizations, and gradient-boosted decision forest / LSTM classifications are performed **entirely in volatile system memory (RAM)** on your local device.
   - No webcam frames, video streams, or biometric hand geometry data are ever transmitted to external servers, cloud services, or third-party APIs.

2. **Local Text-To-Speech (TTS)**:
   - Voice synthesis uses the native system speech synthesizer (`pyttsx3` with Windows SAPI5, macOS NSSpeechSynthesizer, or Linux eSpeak). Speech audio generation is 100% offline with zero cloud API reliance.

3. **No Unsolicited Telemetry or Data Logging**:
   - The application does not collect telemetry, user analytics, tracking cookies, or diagnostic pings.
   - Data collection utilities (`data/collect_data.py`) only save landmark coordinates locally when explicitly and manually triggered by the user.

---

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |

---

## Security Best Practices for Deployment

When running or deploying this application:

- **Webcam Access Permissions**: Ensure only authorized users have access to run webcam video capture on the host system.
- **Environment Variables**: Use `.env.example` as a template for local environment variables. Never commit `.env` or credential files to version control.
- **Model Integrity**: The included pre-trained weights (`models/*.task`, `models/*.pkl`, `models/*.npz`) are verified local artifacts. Do not load unverified model files from untrusted third-party sources.

---

## Reporting a Vulnerability

If you discover a potential security vulnerability, credential leak, or privacy flaw:

1. **Do not open a public GitHub issue.**
2. Please submit a private security advisory through the repository's **Security** tab:
   - Navigate to [GitHub Security Advisories](https://github.com/Krushnakant1979/gesture-detector/security/advisories) and click **Report a vulnerability**.
3. Include detailed steps to reproduce the issue and any relevant system environment details.
4. Vulnerability reports will be acknowledged and reviewed promptly.
