# Wordle Discord Bot

This Discord bot analyzes Wordle screenshots to provide a critique of your gameplay. It uses the Gemini API to parse the image and then calculates the entropy of your guesses to suggest optimal plays.

## Features

- **Wordle Analysis**: The `!wanal` command takes a Wordle screenshot and provides a detailed analysis of your guesses.
- **Optimal Guess Suggestion**: The bot suggests the best possible guess at each step of the game based on information theory.
- **Privacy Focused**: The analysis results are sent to you via DM to avoid spoiling the game for others.

## Prerequisites

- Python 3.7+
- A Discord Bot Token. You can get one by creating a new application on the [Discord Developer Portal](https://discord.com/developers/applications).
- A Gemini API Key. You can get one from [Google AI Studio](https://aistudio.google.com/app/apikey).

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/WordleDCBot.git
   cd WordleDCBot
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install the dependencies:**
   ```bash
   sh install_deps.sh
   ```

4. **Create a `.env` file:**
   Create a file named `.env` in the root of the project and add your Discord token and Gemini API key:
   ```
   DISCORD_TOKEN="your_discord_bot_token"
   GEMINI_API_KEY="your_gemini_api_key"
   ```

## Usage

1. **Run the bot:**
   ```bash
   python bot.py
   ```

2. **Invite the bot to your server:**
   Use the following URL to invite the bot to your Discord server. Replace `YOUR_CLIENT_ID` with your bot's client ID from the Discord Developer Portal.
   `https://discord.com/api/oauth2/authorize?client_id=YOUR_CLIENT_ID&permissions=8&scope=bot`

3. **Use the `!wanal` command:**
   In any channel the bot has access to, use the `!wanal` command and attach a screenshot of your Wordle game. The bot will then send you a DM with the analysis.

   `!wanal`

   You can also provide the target word as an argument if it's not visible in the screenshot:

   `!wanal <target_word>`

## Acknowledgements

- GillesVandewiele's logic for Wordle analysis inspired this project. Check out his work [here](https://github.com/GillesVandewiele/Wordle-Bot)

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.