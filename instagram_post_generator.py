import asyncio
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.teams import DiGraphBuilder, GraphFlow
from autogen_agentchat.messages import TextMessage
from autogen_ext.models.ollama import OllamaChatCompletionClient

async def main():
    """
    Main function to run the Instagram post generation agent workflow.
    """
    # Configure Ollama client
    # Ensure Ollama is running and the model is pulled.
    # User has updated model to "phi4:latest"
    ollama_config = {
        "model": "phi4:latest", # User's chosen model
        # Use "host" for the Ollama server URL, e.g., "http://localhost:11434"
        # "host": "http://localhost:11434", # Optional: Un-comment and set if Ollama is not on default
        # Use "timeout" for the request timeout in seconds
        # "timeout": 120, # Optional: Set request timeout in seconds
    }

    # Initialize OllamaChatCompletionClient
    client_params = {"model": ollama_config["model"]}
    if "host" in ollama_config and ollama_config["host"]: # Check if host is set
        client_params["host"] = ollama_config["host"]
    if "timeout" in ollama_config and ollama_config["timeout"]: # Check if timeout is set
        client_params["timeout"] = ollama_config["timeout"]
    
    model_client = OllamaChatCompletionClient(**client_params)

    # 1. Planner Agent: Takes the topic and plans 3-5 slide themes.
    planner_agent = AssistantAgent(
        name="PlannerAgent",
        model_client=model_client, # Use model_client
        system_message='''You are a helpful AI assistant.
Given a topic, your role is to plan the content for an Instagram post.
The post should have between 3 and 5 slides.
For each slide, define a concise theme or key message.
Output the themes as a numbered list.
Example:
Topic: Benefits of daily exercise
Output:
1. Theme: Improved physical health.
2. Theme: Enhanced mental well-being.
3. Theme: Increased energy levels.
'''
    )

    # 2. Slide Generator Agent: Creates one sentence for each slide based on themes.
    slide_generator_agent = AssistantAgent(
        name="SlideGeneratorAgent",
        model_client=model_client, # Use model_client
        system_message='''You are a helpful AI assistant.
You will receive a list of themes for Instagram slides.
For each theme, write a single, compelling, and concise sentence that would fit well on an Instagram image.
Keep sentences short and impactful.
Output each sentence on a new line, prefixed with "Slide X: ".
Example Input:
1. Theme: Improved physical health.
2. Theme: Enhanced mental well-being.
3. Theme: Increased energy levels.
Example Output:
Slide 1: Boost your physical health with regular movement.
Slide 2: Daily exercise sharpens the mind and lifts your mood.
Slide 3: Feel more energized throughout your day by staying active.
'''
    )

    # 3. Description Generator Agent: Creates the post description and hashtags.
    description_generator_agent = AssistantAgent(
        name="DescriptionGeneratorAgent",
        model_client=model_client, # Use model_client
        system_message='''You are a helpful AI assistant.
You will receive a list of sentences, each representing an Instagram slide.
Your task is to:
1. Write an engaging Instagram post description that summarizes the key messages from the slides.
2. Include a call to action if appropriate for the topic.
3. Add 3-5 relevant hashtags.
4. At the end, list the slide sentences again, each on a new line, prefixed with "Slide X: " for the compiler.

Example Input:
Slide 1: Boost your physical health with regular movement.
Slide 2: Daily exercise sharpens the mind and lifts your mood.
Slide 3: Feel more energized throughout your day by staying active.

Example Output:
Ready to transform your well-being? Incorporating daily exercise can significantly boost your physical health, sharpen your mental focus, and leave you feeling energized all day long! What's your favorite way to stay active? Let us know below! 👇

#DailyExercise #FitnessMotivation #HealthyLifestyle #WellnessJourney #MoveYourBody

Slide 1: Boost your physical health with regular movement.
Slide 2: Daily exercise sharpens the mind and lifts your mood.
Slide 3: Feel more energized throughout your day by staying active.
'''
    )

    # 4. Compiler Agent: Formats the final output.
    compiler_agent = AssistantAgent(
        name="CompilerAgent",
        model_client=model_client, # Use model_client
        system_message='''You are a helpful AI assistant.
You will receive a message containing an Instagram post description, hashtags, and a list of slide sentences.
Your task is to format this information clearly.
The output should be:
INSTAGRAM POST CONTENT:
--- SLIDES ---
[Slide 1 sentence]
[Slide 2 sentence]
...
--- DESCRIPTION ---
[Post description]

[Hashtags]

Ensure the slide sentences are listed one per line under "--- SLIDES ---".
Ensure the description and hashtags are under "--- DESCRIPTION ---".
'''
    )

    # User Proxy Agent: Initiates the chat with the topic.
    user_proxy_agent = UserProxyAgent(
        name="UserProxyAgent",
    )

    # Build the graph
    builder = DiGraphBuilder()
    builder.add_node(planner_agent)
    builder.add_node(slide_generator_agent)
    builder.add_node(description_generator_agent)
    builder.add_node(compiler_agent)

    # Define the flow: Planner -> SlideGenerator -> DescriptionGenerator -> Compiler
    builder.add_edge(planner_agent, slide_generator_agent)
    builder.add_edge(slide_generator_agent, description_generator_agent)
    builder.add_edge(description_generator_agent, compiler_agent)

    # Set the entry point
    builder.set_entry_point(planner_agent)

    graph = builder.build()

    # Create the GraphFlow
    flow = GraphFlow(
        participants=builder.get_participants(),
        graph=graph
        # Optionally, add termination_condition, max_turns, etc.
    )

    # Initiate the chat
    topic = input("Enter the topic for your Instagram post: ")
    if not topic:
        topic = "The future of renewable energy" # Default topic
        print(f"No topic entered, using default: {topic}")

    print("\nGenerating Instagram post content...\n")

    # The initial message to the graph should be to the entry point agent (planner_agent)
    # By passing the topic string directly to flow.run(task=topic),
    # GraphFlow will handle creating an initial message (typically UserMessage with source="user")
    # and directing it to the entry point.
    # initial_message = TextMessage(content=topic, source=user_proxy_agent.name, destination=planner_agent.name)

    # Run the graph
    # The `run` method needs a `task` argument which can be a BaseChatMessage or a string.
    # Passing the topic string directly is the standard way for GraphFlow.
    final_response_task_result = await flow.run(task=topic) # Pass the topic string directly

    output_content = None
    if final_response_task_result and final_response_task_result.messages:
        # Iterate in reverse to find the last message from the compiler_agent
        for message in reversed(final_response_task_result.messages):
            if hasattr(message, 'source') and message.source == compiler_agent.name:
                if hasattr(message, 'content') and message.content:
                    output_content = str(message.content) # Ensure it's a string
                    break
        
        if output_content:
            print("\n--- INSTAGRAM POST CONTENT ---")
            print(output_content)
            print("--- END OF POST ---\n")

            # Save to file
            try:
                # Sanitize topic for filename
                safe_topic_chars = []
                for char_in_topic in topic:
                    if char_in_topic.isalnum() or char_in_topic == ' ':
                        safe_topic_chars.append(char_in_topic)
                
                safe_topic = "".join(safe_topic_chars).strip().replace(' ', '_').lower()
                
                if not safe_topic: # Handle case where topic becomes empty after sanitization
                    safe_topic = "untitled_post"
                filename = f"{safe_topic}_instagram_post.txt"
                
                with open(filename, "w", encoding="utf-8") as f:
                    f.write(output_content)
                print(f"Output also saved to: {filename}")
            except Exception as e:
                print(f"Error saving output to file: {e}")

        else: # No content found from compiler_agent or content was empty
            print("Could not find valid content from the CompilerAgent in the chat history.")
            print("Dumping all messages from TaskResult for debugging:")
            for i, msg in enumerate(final_response_task_result.messages):
                source_name = getattr(msg, 'source', 'N/A')
                msg_type = type(msg).__name__
                msg_content = getattr(msg, 'content', '<NO CONTENT ATTRIBUTE>')
                print(f"Message {i}: Source='{source_name}', Type='{msg_type}', Content='{msg_content}'")
                if msg_content == '<NO CONTENT ATTRIBUTE>':
                    print(f"  Full Message Object: {msg}")


    else: # TaskResult is None or has no messages
        print("No response or messages received from the graph execution.")
        print("Full TaskResult object (if any):")
        print(final_response_task_result)

    # Close the model client
    await model_client.close()

if __name__ == "__main__":
    asyncio.run(main()) 