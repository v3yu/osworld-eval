"""Simplified observation processors for image and text observations"""
import numpy as np
import numpy.typing as npt
from beartype import beartype
from PIL import Image, ImageDraw
from playwright.sync_api import CDPSession, Page
from typing import TypedDict

from .utils import Observation, png_bytes_to_numpy


class ObservationMetadata(TypedDict):
    """Simple observation metadata for compatibility with the original MMInA structure"""
    pass


class ObservationProcessor:
    """Base class for observation processors"""
    def process(self, page: Page, client: CDPSession) -> Observation:
        raise NotImplementedError


class SimpleTextObservationProcessor(ObservationProcessor):
    """Simple text observation processor that returns empty text for now"""
    
    def __init__(self, args=None):
        self.args = args

    @beartype
    def process(self, page: Page, client: CDPSession) -> str:
        """Process the page and return empty text (placeholder for future text processing)"""
        # TODO: Add self-reflection, history description, etc. here later
        return ""


class SimpleImageObservationProcessor(ObservationProcessor):
    """Simple image observation processor that only handles screenshots"""
    
    def __init__(self, args=None):
        self.args = args
        self.interaction_coords = None  # Store the current interaction coordinates

    def set_interaction_point(self, x: float, y: float):
        """Set the interaction point coordinates for drawing a red circle"""
        self.interaction_coords = (x, y)

    def set_interaction_point_from_action(self, action):
        """Set interaction point from action coordinates"""
        if action and 'action_inputs' in action:
            start_box = action['action_inputs'].get('start_box')
            if start_box:
                try:
                    if isinstance(start_box, str):
                        coords = eval(start_box)
                    else:
                        coords = start_box
                    
                    if len(coords) >= 2:
                        if len(coords) == 4:
                            # If it's a bounding box, calculate center
                            x1, y1, x2, y2 = coords
                            x = (x1 + x2) / 2
                            y = (y1 + y2) / 2
                        else:
                            # If it's already coordinates
                            x, y = coords[0], coords[1]
                        
                        self.set_interaction_point(x, y)
                        return True
                except Exception as e:
                    print(f"Error parsing action coordinates: {e}")
        return False

    def clear_interaction_point(self):
        """Clear the interaction point"""
        self.interaction_coords = None

    @beartype
    def process(self, page: Page, client: CDPSession) -> npt.NDArray[np.uint8]:
        """Process the page and return a screenshot as numpy array"""
        try:
            screenshot = png_bytes_to_numpy(page.screenshot())
        except Exception as e:
            print(f"Screenshot failed, trying to wait for page load: {e}")
            try:
                # Wait for page to load with a shorter timeout
                page.wait_for_event("load", timeout=10000)  # 10 second timeout
                screenshot = png_bytes_to_numpy(page.screenshot())
            except Exception as load_error:
                screenshot = np.zeros((720, 1280, 3), dtype=np.uint8)
        
        # Add red circle if interaction coordinates are set
        if self.interaction_coords is not None:
            try:
                # Convert numpy array to PIL Image
                image = Image.fromarray(screenshot)
                
                # Create a drawing context
                draw = ImageDraw.Draw(image)
                
                # Get coordinates
                x, y = self.interaction_coords
                
                # Draw red circle
                circle_radius = 20
                draw.ellipse((x - circle_radius, y - circle_radius, 
                            x + circle_radius, y + circle_radius), 
                            outline=(255, 0, 0), width=5)
                
                # Convert back to numpy array
                screenshot = np.array(image)
                
            except Exception as e:
                print(f"Error drawing interaction circle: {e}")
        
        return screenshot
