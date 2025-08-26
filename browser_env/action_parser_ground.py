import re
from PIL import Image
import io
import torch
from .actions import ActionTypes


def get_coords_from_grounding_model(action, element, grounding_model, tokenizer, image):
    if element == '' or element is None:
        coords = re.sub(r'[^\d\s,.-]', '', action).strip()
        coords = re.split(r'[,\s]+', coords)
        try:
            return [float(coords[0]), float(coords[1])]
        except:
            return [0, 0]
    
    instruct = "You are a grounding model, given the screenshot and the target element description, you need to identify the coordinates of the given element and return them in the format of click(point='<point>x1 y1</point>')."
    query = "Target element description: " + element + "\nWhat's the coordinates of the target element in the screenshot? You should return as click(point='<point>x1 y1</point>')"
    
    # Check if grounding_model is an OpenAI client (has chat.completions)
    if hasattr(grounding_model, 'chat') and hasattr(grounding_model.chat, 'completions'):
        # Use OpenAI client format
        try:
            response = grounding_model.chat.completions.create(
                model="ByteDance-Seed/UI-TARS-1.5-7B",  # Your grounding model
                messages=[
                    {"role": "system", "content": instruct},
                    {"role": "user", "content": [
                        {"type": "text", "text": query},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image}"}}
                    ]}
                ],
                temperature=0.1,
                max_tokens=512
            )
            
            response_text = response.choices[0].message.content
            print(f"Grounding model response: {response_text}")
            
        except Exception as e:
            print(f"Error calling grounding model: {e}")
            return [0, 0]
    elif tokenizer is not None:
        # Use local model (original code)
        message = [{"role": "system", "content": [{"type": "text", "text": instruct}]},
               {"role": "user", "content": [{"type": "text", "text": query}, {"type": "image", "image": image}]}]
        inputs = tokenizer.apply_chat_template(
                                message,
                                add_generation_prompt=True,
                                tokenize=True,
                                return_tensors="pt",
                                return_dict=True,
                            ).to(grounding_model.device)
        with torch.no_grad():
            outputs = grounding_model.generate(**inputs)
            outputs = outputs[:, inputs["input_ids"].shape[1]:]
            response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"Grounding model response: {response_text}")
    else:
        # No tokenizer available, return default coordinates
        print("No tokenizer available for grounding model")
        return [0, 0]
    coords = re.sub(r'[^\d\s,.-]', '', response_text).strip()
    coords = re.split(r'[,\s]+', coords)
    coordinates = []
    if 'x1' in response_text:
        coords[0] = coords[0][1:]
    if 'y1' in response_text:
        coords[1] = coords[1][1:]
        print(f"Extracted coordinates with x1: {coords}")
    for coord in coords:
        if coord:
            try:
                coordinates.append(float(coord))
            except:
                print(f"Invalid coordinate value: {coord}")
                coordinates.append(0)
    coordinates = coordinates[:2]
    print(f"Extracted coordinates: {coordinates}")
    return coordinates


def execute_pixel_action(responses, page, image_processor=None, observation=None, args=None):
    """
    Execute actions using Playwright based on the model's output
    
    Args:
        page: Playwright page object
        responses: Dictionary or list of dictionaries containing action data
        image_processor: Optional ImageObservationProcessor for visualization
        observation: Optional observation dict to update with visualization
    
    Returns:
        True if actions were executed, "DONE" if finished action was encountered
    """
    if isinstance(responses, dict):
        responses = [responses]

    for response_id, response in enumerate(responses):
        # Extract action data from new action structure
        action_type = response.get("action_type")
        
        print(f"Executing {action_type} action...")

        # Set interaction point for visualization if image_processor is available
        if image_processor is not None:
            image_processor.set_interaction_point_from_action(response)

        if action_type == ActionTypes.CLICK:
            # Handle click action with description
            description = response.get("description", "")
            reasoning = response.get("reasoning", "")
            
            # Must use grounding model to get coordinates
            if not args or not hasattr(args, 'grounding_model') or not args.grounding_model:
                print(f"Error: grounding_model is required for click action but not available")
                continue
                
            coords = get_coords_from_grounding_model(
                "", description, args.grounding_model, None, observation.get("image", None)
            )
            if coords and len(coords) >= 2:
                page.mouse.click(coords[0], coords[1])
                print(f"Clicked at coordinates {coords} for element: {description}")
            else:
                print(f"Could not determine coordinates for: {description}")

        elif action_type == ActionTypes.TYPE:
            # Handle type action with field description
            text = response.get("text", "")
            field_description = response.get("field_description", "")
            reasoning = response.get("reasoning", "")
            
            # Must use grounding model to get coordinates
            if not args or not hasattr(args, 'grounding_model') or not args.grounding_model:
                print(f"Error: grounding_model is required for type action but not available")
                continue
                
            coords = get_coords_from_grounding_model(
                "", field_description, args.grounding_model, None, observation.get("image", None)
            )
            if coords and len(coords) >= 2:
                # Click at the coordinates first to focus the input field
                page.mouse.click(coords[0], coords[1])
                # Clear the field and type the text
                page.keyboard.press("Control+a")
                page.keyboard.press("Delete")
                page.keyboard.type(text)
                print(f"Typed '{text}' at coordinates {coords} for field: {field_description}")
            else:
                print(f"Could not determine coordinates for field: {field_description}")

        elif action_type == ActionTypes.SCROLL:
            # Handle scroll action
            direction = response.get("direction", "down")
            reasoning = response.get("reasoning", "")
            
            if direction == "up":
                page.mouse.wheel(0, -500)
            elif direction == "down":
                page.mouse.wheel(0, 500)
            elif direction == "left":
                page.mouse.wheel(-500, 0)
            elif direction == "right":
                page.mouse.wheel(500, 0)
            
            print(f"Scrolled {direction}")

        elif action_type == ActionTypes.WAIT:
            # Handle wait action
            seconds = response.get("seconds", 2.0)
            reasoning = response.get("reasoning", "")
            
            page.wait_for_timeout(int(seconds * 1000))
            print(f"Waited for {seconds} seconds")

        elif action_type == ActionTypes.KEY_PRESS:
            # Handle key press action
            key_comb = response.get("key_comb", "enter")
            reasoning = response.get("reasoning", "")
            
            page.keyboard.press(key_comb)
            print(f"Pressed key: {key_comb}")

        elif action_type == ActionTypes.GOTO_URL:
            # Handle goto URL navigation
            url = response.get("url", "")
            if url:
                try:
                    page.goto(url)
                    if 'twitter' in url:
                        page.click('input[name="text"][autocomplete="username"]')
                        page.fill('input[name="text"][autocomplete="username"]', "8582149622")
                        page.click('button[class="css-175oi2r r-sdzlij r-1phboty r-rs99b7 r-lrvibr r-ywje51 r-184id4b r-13qz1uu r-2yi16 r-1qi8awa r-3pj75a r-1loqt21 r-o7ynqc r-6416eg r-1ny4l3l"]')
                        page.wait_for_timeout(1000)
                        page.fill('input[name="password"][autocomplete="current-password"]', "3129028a.")
                        page.press('input[name="password"][autocomplete="current-password"]', "Enter")
                        page.wait_for_timeout(1000)
                    
                    print(f"Navigated to URL: {url}")
                except Exception as e:
                    print(f"Failed to navigate to {url}: {e}")
            else:
                print("No URL provided for GOTO_URL action")

        elif action_type == ActionTypes.STOP:
            # Handle stop action
            answer = response.get("answer", "Task completed")
            reasoning = response.get("reasoning", "")
            
            print(f"Task finished: {answer}")
            return "DONE"

        else:
            print(f"Unknown action type: {action_type}")
            
        # Clear interaction point for visualization
        if image_processor is not None:
            image_processor.clear_interaction_point()
            
        # Add a small delay between actions
        if response_id < len(responses) - 1:
            page.wait_for_timeout(1000)  # 1 second delay
            
    return page


def get_action_description(action) -> str:
    """Generate the text version of the predicted actions to store in action history for prompt use.
    Updated to work with the new action structure and ActionTypes enum."""

    from .actions import ActionTypes

    action_type = action.get("action_type", "unknown")
    
    if action_type == ActionTypes.CLICK:
        description = action.get("description", "")
        reasoning = action.get("reasoning", "")
        action_str = f"click(description='{description}')"
        if reasoning:
            action_str += f" - {reasoning}"
    
    elif action_type == ActionTypes.TYPE:
        text = action.get("text", "")
        field_description = action.get("field_description", "")
        reasoning = action.get("reasoning", "")
        # Escape content for proper display
        text = text.replace("'", "\\'").replace('"', '\\"').replace("\n", "\\n")
        action_str = f"type(text='{text}', field='{field_description}')"
        if reasoning:
            action_str += f" - {reasoning}"
    
    elif action_type == ActionTypes.SCROLL:
        direction = action.get("direction", "down")
        reasoning = action.get("reasoning", "")
        action_str = f"scroll(direction='{direction}')"
        if reasoning:
            action_str += f" - {reasoning}"
    
    elif action_type == ActionTypes.WAIT:
        seconds = action.get("seconds", 2.0)
        reasoning = action.get("reasoning", "")
        action_str = f"wait(seconds={seconds})"
        if reasoning:
            action_str += f" - {reasoning}"
    
    elif action_type == ActionTypes.KEY_PRESS:
        key_comb = action.get("key_comb", "enter")
        reasoning = action.get("reasoning", "")
        action_str = f"press_key(key='{key_comb}')"
        if reasoning:
            action_str += f" - {reasoning}"

    elif action_type == ActionTypes.GOTO_URL:
        url = action.get("url", "")
        reasoning = action.get("reasoning", "")
        action_str = f"goto_url(url='{url}')"
        if reasoning:
            action_str += f" - {reasoning}"

    elif action_type == ActionTypes.STOP:
        answer = action.get("answer", "")
        reasoning = action.get("reasoning", "")
        # Escape content for proper display
        answer = answer.replace("'", "\\'").replace('"', '\\"').replace("\n", "\\n")
        action_str = f"finished(answer='{answer}')"
        if reasoning:
            action_str += f" - {reasoning}"
    
    else:
        # For any other action types, use the new structure if available
        if "reasoning" in action:
            action_str = f"{action_type}({', '.join([f'{k}={v}' for k, v in action.items() if k not in ['action_type', 'reasoning']])}) - {action.get('reasoning', '')}"
        else:
            # Fallback to old structure
            action_str = f"{action_type}({', '.join([f'{k}={v}' for k, v in action.get('action_inputs', {}).items()])})"
        
    return action_str



