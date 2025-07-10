# modifies the anthropic chat completion create call to make sure the resulting output is a valid pydantic model

import copy
import json
import anthropic
from pydantic import ValidationError

from openai import OpenAI

from dotenv import load_dotenv

load_dotenv()

class PydanticAdaptorOpenRouter:

    def __init__(self, openai_client=None, openai_api_key=None) -> None:
        if openai_client is not None:
            self.openai_client = openai_client
        else:
            if openai_api_key is None:
                self.openai_client = OpenAI(base_url="https://openrouter.ai/api/v1")
            else:
                self.openai_client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=openai_api_key)
    
    @property
    def chat(self):
        return self
    
    @property
    def completions(self):
        return self
    

    def _pydantic_model_to_tool_definition(self, pydantic_model):
        pydantic_json_schema = pydantic_model.model_json_schema()
        tool_name = pydantic_json_schema.pop("title")
        tool_description = pydantic_json_schema.pop("description", "")

        function = {
            "name": tool_name,
            "description": tool_description,
            "parameters": pydantic_json_schema
        }

        tool_definition = {
            "type": "function",
            "function": function
        }

        return tool_definition


    def _parse_tool_call_response(self, chat_completion_response):
        # assuming n is always 1
        response = chat_completion_response.choices[0]
        # if response.finish_reason != "tool_calls":
        #     raise ValueError(f"The model completion did not return a tool use block. Here is the full response - {chat_completion_response}")

        tool_calls = response.message.tool_calls
        if tool_calls is None or len(tool_calls) <= 0:
            raise ValueError(f"The model completion did not return a tool use block. Here is the full response - {chat_completion_response}")

        tool_call = tool_calls[0]

        parsed_response = {
            'tool_call_id': tool_call.id,
            'tool_call_name': tool_call.function.name,
            'tool_call_arguments': tool_call.function.arguments
        }

        return parsed_response


    def _add_tool_call_retry_messages(self, messages_list, parsed_tool_call_response, error_details):
        assistant_tool_call_message = {
            "role": "assistant",
            "tool_calls": [{
                "type": "function",
                "id": parsed_tool_call_response["tool_call_id"],
                "function": {
                    "name": parsed_tool_call_response["tool_call_name"],
                    "arguments": parsed_tool_call_response["tool_call_arguments"]
                }  
            }]
        }

        tool_response_message = {
            "role": "tool",
            "tool_call_id": parsed_tool_call_response["tool_call_id"],
            "content": [{
                "type": "tool_result",
                "tool_use_id": parsed_tool_call_response["tool_call_id"],
                "content": [{
                    "type": "text",
                    "text": error_details
                }]
            }]
        }

        messages_list.append(assistant_tool_call_message)
        messages_list.append(tool_response_message)
        
        return messages_list


    #TODO assumes only a single tool call
    def create(self, *, pydantic_model, num_retries=5, **kwargs):

        if "tools" in kwargs:
            raise ValueError(
                "you cannot pass `tools` explicitly. The `PydanticAdaptorAnthropic` class uses the tool use functionality of anthropic to guarantee the response adheres to the given pydantic model."
                )
        if "tool_choice" in kwargs:
            raise ValueError(
                "you cannot pass `tool_choice` explicitly. The `PydanticAdaptorAnthropic` class uses the tool use functionality of anthropic to guarantee the response adheres to the given pydantic model."
                )


        # convert pydantic model into anthropic tools params
        tool_definition = self._pydantic_model_to_tool_definition(pydantic_model=pydantic_model)
        tools = [tool_definition]
        tool_choice = {
            "type": "function",
            "function": {
                "name": tool_definition["function"]["name"] 
            }
        }

        formatted_kwargs = copy.deepcopy(kwargs)
        formatted_kwargs["tools"] = tools
        formatted_kwargs["tool_choice"] = tool_choice

        # validate the response
        fit_pydantic_model = None
        validation_error_details = None
        parsed_tool_call_response = None
        curr_try = 0
        #NOTE total allowed tries is initial_try(1) + num_retries
        while (not fit_pydantic_model and curr_try <= num_retries):
            try:
                # if you are in a retry, add previous response and errors to the messages array
                if curr_try > 1:
                    updated_message = self._add_tool_call_retry_messages(messages_list=formatted_kwargs["messages"], parsed_tool_call_response=parsed_tool_call_response, error_details=validation_error_details)
                    formatted_kwargs["messages"] = updated_message        

                # make the anthropic call
                chat_completions_response = self.openai_client.chat.completions.create(
                    **formatted_kwargs
                )            

                parsed_tool_call_response = self._parse_tool_call_response(chat_completions_response)
                
                gen_tool_call_input_string = parsed_tool_call_response["tool_call_arguments"]
                gen_tool_call_input = json.loads(gen_tool_call_input_string)
                fit_pydantic_model = pydantic_model.model_validate(gen_tool_call_input)

            except json.JSONDecodeError as e:
                # handle case where tool call arguments are not valid JSON
                validation_error_details = str(e)
            
            except ValidationError as e:
                # get the validation errors and add it as new validation context for the next retry
                validation_error_details = e.json()
            
            finally:
                curr_try += 1

        if not fit_pydantic_model:
            raise RuntimeError(f"Unable to get a structured output from the model within max retries - {num_retries}")

        return fit_pydantic_model


if __name__ == "__main__":
    
    # from dotenv import load_dotenv
    from pydantic import BaseModel, Field
    import json

    # load_dotenv()

    class NextStepAnswer(BaseModel):
        """This defines the next step the user needs to take given their final objective and the
        current screen's contents.
        """

        step_description: str = Field(description="one line textual description of the next step")
    

    adaptor = PydanticAdaptorOpenRouter()

    content = [{"type": "text", "text": "Help the user achieve their goal."}]

    curr_message = {
        "role": "user",
        "content": content
    }

    message_history = [curr_message]

    response = adaptor.chat.completions.create(
        pydantic_model=NextStepAnswer,
        num_retries=1,
        # model="meta-llama/llama-4-maverick",
        model="openai/gpt-4o-mini",
        messages=message_history,
        max_tokens=4096,
        stream=False
    )

    print(f"{response=}")
