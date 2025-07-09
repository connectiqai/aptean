# modifies the anthropic chat completion create call to make sure the resulting output is a valid pydantic model

import copy
import anthropic
from pydantic import ValidationError

class PydanticAdaptorAnthropic:

    def __init__(self, anthropic_client=None, anthropic_api_key=None) -> None:
        if anthropic_client is not None:
            self.anthropic_client = anthropic_client
        else:
            if anthropic_api_key is None:
                self.anthropic_client = anthropic.Anthropic()
            else:
                self.anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key)
    
    @property
    def messages(self):
        return self
    

    def _pydantic_model_to_tool_definition(self, pydantic_model):
        pydantic_json_schema = pydantic_model.model_json_schema()
        tool_name = pydantic_json_schema.pop("title")
        tool_description = pydantic_json_schema.pop("description", "")

        tool_definition = {
            "name": tool_name,
            "description": tool_description,
            "input_schema": pydantic_json_schema
        }

        return tool_definition


    def _parse_tool_call_response(self, chat_completion_response):

        content_list = chat_completion_response.content
        #NOTE assume a single response
        content_response = content_list[0]

        if content_response.type != "tool_use":
            raise ValueError(f"The anthropic completion did not return a tool use block. Here is the full response - {chat_completion_response}")
        tool_call_id = content_response.id
        tool_call_name = content_response.name
        tool_call_input = content_response.input
            

        parsed_response = {
            "tool_call_id": tool_call_id,
            "tool_call_name": tool_call_name,
            "tool_call_input": tool_call_input 
        }

        return parsed_response


    def _add_tool_call_retry_messages(self, messages_list, parsed_tool_call_response, error_details):
        assistant_tool_call_message = {
            "role": "assistant",
            "content": [{
                "type": "tool_use",
                "id": parsed_tool_call_response["tool_call_id"],
                "name": parsed_tool_call_response["tool_call_name"],
                "input": parsed_tool_call_response["tool_call_input"]
            }]
        }

        tool_response_message = {
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": parsed_tool_call_response["tool_call_id"],
                "is_error": True,
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
            "type": "tool",
            "name": tool_definition["name"]
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
                chat_completions_response = self.anthropic_client.messages.create(
                    **formatted_kwargs
                )            

                parsed_tool_call_response = self._parse_tool_call_response(chat_completions_response)
                
                gen_tool_call_input = parsed_tool_call_response["tool_call_input"]
                fit_pydantic_model = pydantic_model.model_validate(gen_tool_call_input)

            except ValidationError as e:
                # get the validation errors and add it as new validation context for the next retry
                validation_error_details = e.json()

            finally:
                curr_try += 1

        if not fit_pydantic_model:
            raise RuntimeError(f"Unable to get a structured output from the model within max retries - {num_retries}")

        return fit_pydantic_model


if __name__ == "__main__":
    
    from dotenv import load_dotenv
    from pydantic import BaseModel, Field
    import json

    load_dotenv()

    class NextStepAnswer(BaseModel):
        """This defines the next step the user needs to take given their final objective and the
        current screen's contents.
        """

        step_description: str = Field(description="one line textual description of the next step")
    

    adaptor = PydanticAdaptorAnthropic()

    content = [{"type": "text", "text": "Help the user achieve their goal."}]

    curr_message = {
        "role": "user",
        "content": content
    }

    message_history = [curr_message]

    response = adaptor.messages.create(
        pydantic_model=NextStepAnswer,
        num_retries=1,
        model="claude-3-5-sonnet-20240620",
        messages=message_history,
        max_tokens=4096,
        stream=False
    )

    print(f"{response=}")
