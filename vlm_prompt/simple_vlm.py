import os
import sys
import requests
from PIL import Image
import io
import time
import json

print("Sports Image Analysis Tool")
print("==========================")

def ensure_packages():
    """Install required packages if necessary."""
    try:
        from PIL import Image
        import requests
        print("Required packages already installed.")
    except ImportError:
        import subprocess
        print("Installing required packages...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "pillow", "requests"
        ])
        print("Packages installed successfully!")

# Ensure we have required packages
ensure_packages()

def get_image_stats(image_path):
    """Get basic statistics about an image."""
    try:
        image = Image.open(image_path)
        width, height = image.size
        format_name = image.format
        mode = image.mode
        
        # Calculate average brightness - handle RGBA by converting to RGB first
        if mode == 'RGBA':
            # Convert RGBA to RGB
            rgb_image = Image.new("RGB", image.size, (255, 255, 255))
            rgb_image.paste(image, mask=image.split()[3])  # Use alpha as mask
            image_for_analysis = rgb_image
        elif mode == 'RGB':
            image_for_analysis = image
        else:
            image_for_analysis = image.convert('RGB')  # Try to convert anyway
            
        # Get color information
        colors = get_dominant_colors(image_for_analysis)
            
        gray = image_for_analysis.convert('L')
        pixels = list(gray.getdata())
        avg_brightness = sum(pixels) / len(pixels)
        brightness_percent = avg_brightness / 255 * 100
            
        stats = {
            "format": format_name,
            "width": width,
            "height": height,
            "mode": mode,
            "brightness": f"{brightness_percent:.1f}%",
            "colors": colors
        }
        
        return stats, image
    except Exception as e:
        print(f"Error analyzing image: {e}")
        return None, None

def get_dominant_colors(image, num_colors=5):
    """Get the dominant colors in the image."""
    # Resize image for faster processing
    small_img = image.resize((100, 100))
    
    # Get all pixels
    pixels = list(small_img.getdata())
    
    # Count colors (simplified approach)
    color_counts = {}
    for pixel in pixels:
        # Simplify colors by rounding to nearest 30
        if isinstance(pixel, tuple) and len(pixel) >= 3:
            simple_pixel = (round(pixel[0]/30)*30, round(pixel[1]/30)*30, round(pixel[2]/30)*30)
            color_counts[simple_pixel] = color_counts.get(simple_pixel, 0) + 1
    
    # Sort by frequency
    sorted_colors = sorted(color_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Get top colors and their names
    top_colors = []
    for i, (color, count) in enumerate(sorted_colors[:num_colors]):
        percentage = count / len(pixels) * 100
        if percentage > 5:  # Only include colors that make up more than 5%
            color_name = get_color_name(color)
            top_colors.append({
                "rgb": color,
                "name": color_name,
                "percentage": percentage
            })
    
    return top_colors

def get_color_name(rgb):
    """Convert RGB to a color name."""
    r, g, b = rgb
    
    # Simple color naming
    if max(r, g, b) < 40:
        return "black"
    if min(r, g, b) > 200:
        return "white"
    
    if r > g + 50 and r > b + 50:
        if r > 200:
            return "bright red"
        return "red"
    if g > r + 50 and g > b + 50:
        if g > 200:
            return "bright green"
        return "green"
    if b > r + 50 and b > g + 50:
        if b > 200:
            return "bright blue"
        return "blue"
    
    if r > 180 and g > 180 and b < 100:
        return "yellow"
    if r > 180 and g < 100 and b > 180:
        return "purple"
    if r < 100 and g > 180 and b > 180:
        return "cyan"
    if r > 180 and g > 120 and b < 100:
        return "orange"
    if r > 180 and g < 100 and b < 100:
        return "red"
    if r < 100 and g > 180 and b < 100:
        return "green"
    if r < 100 and g < 100 and b > 180:
        return "blue"
    
    # If we get here, it's a mixed/gray color
    avg = (r + g + b) / 3
    if avg > 200:
        return "light gray"
    if avg > 100:
        return "gray"
    return "dark gray"

def analyze_sports_image(image_path, prompt):
    """Analyze a sports image based on the given prompt."""
    stats, image = get_image_stats(image_path)
    if not stats:
        return "Failed to analyze image."
    
    # Process the prompt to determine what to analyze
    prompt_lower = prompt.lower()
    
    results = []
    
    # Team colors analysis
    if "color" in prompt_lower and ("team" in prompt_lower or "wear" in prompt_lower):
        team_colors = analyze_team_colors(stats, image)
        results.append(f"Team Colors: {team_colors}")
    
    # Player with ball analysis
    if "player" in prompt_lower and "ball" in prompt_lower:
        player_analysis = analyze_player_with_ball(stats, image)
        results.append(f"Player with Ball: {player_analysis}")
    
    # If no specific analyses were performed, add a general description
    if not results:
        results.append(f"General Analysis: This appears to be a {stats['width']}x{stats['height']} image that may contain sports content.")
        
        # Add color information
        if stats['colors']:
            color_desc = ", ".join([f"{c['name']} ({c['percentage']:.1f}%)" for c in stats['colors']])
            results.append(f"Dominant Colors: {color_desc}")
    
    return "\n".join(results)

def analyze_team_colors(stats, image):
    """Analyze team colors in the image."""
    colors = stats['colors']
    
    if not colors:
        return "Could not determine team colors."
    
    # Group colors that might represent different teams
    team_colors = []
    
    # Filter out black/white/gray as they are often not team colors
    filtered_colors = [c for c in colors if not any(x in c['name'] for x in ['black', 'white', 'gray'])]
    
    if len(filtered_colors) >= 2:
        # Likely two teams with different colors
        team1 = filtered_colors[0]['name']
        team2 = filtered_colors[1]['name']
        return f"There appear to be two teams: one wearing {team1} and the other wearing {team2}."
    elif len(filtered_colors) == 1:
        # Only one prominent color
        return f"One team appears to be wearing {filtered_colors[0]['name']}."
    else:
        # Fall back to including black/white
        if len(colors) >= 2:
            team1 = colors[0]['name']
            team2 = colors[1]['name']
            return f"Teams appear to be wearing {team1} and {team2}."
        elif len(colors) == 1:
            return f"One team appears to be wearing {colors[0]['name']}."
    
    return "Could not determine team colors clearly."

def analyze_player_with_ball(stats, image):
    """Attempt to determine which player has the ball."""
    # This is a simplified approximation as we can't do true object detection
    return "Without advanced computer vision capabilities, I cannot determine which specific player has the ball. This would require object detection technology."

def process_image(image_path, prompt=None):
    """Process an image file and return results based on the prompt."""
    print(f"Processing image: {image_path}")
    
    # Get basic image stats
    stats, image = get_image_stats(image_path)
    if not stats:
        return "Failed to analyze image."
    
    result = ""
    
    # If there's a specific prompt, analyze accordingly
    if prompt:
        # Identify sports-specific questions
        sports_related = any(term in prompt.lower() for term in ["team", "player", "ball", "sport", "game", "match"])
        
        if sports_related:
            result += analyze_sports_image(image_path, prompt) + "\n\n"
        else:
            result += f"Prompt: {prompt}\n"
            result += "Note: For sports analysis, try asking about team colors or players.\n\n"
    
    # Add general image information
    result += f"Image Details:\n"
    result += f"- Format: {stats['format']}\n"
    result += f"- Size: {stats['width']}x{stats['height']} pixels\n"
    result += f"- Color Mode: {stats['mode']}\n"
    result += f"- Average Brightness: {stats['brightness']}\n\n"
    
    # Add color analysis
    if stats['colors']:
        result += "Color Analysis:\n"
        for color in stats['colors']:
            result += f"- {color['name'].capitalize()}: {color['percentage']:.1f}%\n"
    
    return result

if __name__ == "__main__":
    # Define default image path
    image_path = os.path.join("samples", "sample_image.png")
    prompt = None
    
    # Check command line arguments for image path
    if len(sys.argv) > 1:
        if os.path.exists(sys.argv[1]):
            image_path = sys.argv[1]
        else:
            # If not a file, treat as a prompt
            prompt = sys.argv[1]
    
    # If image doesn't exist, ask for path
    if not os.path.exists(image_path):
        print(f"Image not found at {image_path}")
        print("Please enter the full path to an image file:")
        user_input = input().strip()
        
        if os.path.exists(user_input):
            image_path = user_input
        else:
            print(f"Error: Image not found at {user_input}")
            sys.exit(1)
    
    # If no prompt given yet, ask for one
    if not prompt:
        print("\nEnter your question about the image (press Enter to skip):")
        prompt = input().strip()
    
    # Process the image
    print(f"Analyzing image: {image_path}")
    if prompt:
        print(f"With prompt: {prompt}")
    
    start_time = time.time()
    result = process_image(image_path, prompt)
    elapsed = time.time() - start_time
    
    # Display the results
    print("\n" + "="*50)
    print(result)
    print(f"Processing completed in {elapsed:.2f} seconds.")
    print("="*50)
    
    # Suggest sports-specific prompts if none given
    if not prompt:
        print("\nFor sports images, try these prompts:")
        print("- Which player has the ball?")
        print("- What colors do the teams wear?") 