# AI Instagram Post Generator

This script uses a multi-agent system built with AutoGen and powered by a local Ollama LLM to generate Instagram post content (slides and description) based on a user-provided topic.

## Features

-   Generates 3-5 slide themes for an Instagram post.
-   Creates a concise sentence for each slide.
-   Writes an engaging post description with hashtags.
-   Formats the output clearly.
-   Saves the generated content to a text file named after the topic.

## Prerequisites

-   Python 3.9+
-   Conda (or Miniconda) installed.
-   Ollama installed and running. (Download from [https://ollama.com/](https://ollama.com/))

## Setup Instructions

1.  **Clone the Repository / Download Files:**
    If this script is part of a repository, clone it. Otherwise, ensure `instagram_post_generator.py` and `requirements.txt` are in the same directory.

2.  **Create and Activate Conda Environment:**
    Open your terminal and run the following commands:
    ```bash
    conda create -n ig_post_agent python=3.12 -y
    conda activate ig_post_agent
    ```

3.  **Install Dependencies:**
    With the `ig_post_agent` environment activated, install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up Ollama Model:**
    -   Ensure your Ollama application is running.
    -   Pull the model specified in the script (default is `phi4:latest`). You can do this by running:
        ```bash
        ollama pull phi4:latest
        ```
        If you wish to use a different model, update the `ollama_config` in `instagram_post_generator.py`.

## Running the Script

1.  **Activate Conda Environment (if not already active):**
    ```bash
    conda activate ig_post_agent
    ```

2.  **Run the Script:**
    ```bash
    python instagram_post_generator.py
    ```

3.  **Enter Topic:**
    The script will prompt you to "Enter the topic for your Instagram post:". Type your desired topic and press Enter. If you don't enter a topic, it will use a default ("The future of renewable energy").

4.  **View Output:**
    -   The generated Instagram post content will be printed to the terminal.
    -   The content will also be saved to a text file in the same directory (e.g., `your_topic_instagram_post.txt`).

## Customization

-   **Ollama Model:** You can change the Ollama model by modifying the `model` key in the `ollama_config` dictionary within `instagram_post_generator.py`.
    ```python
    ollama_config = {
        "model": "your-ollama-model:tag", # Change this
        # ... other optional parameters like host and timeout
    }
    ```
-   **Ollama Server URL:** If your Ollama instance is not running on the default `http://localhost:11434`, you can un-comment and set the `host` parameter in `ollama_config`.
-   **Request Timeout:** You can adjust the request timeout by un-commenting and setting the `timeout` parameter in `ollama_config`. 