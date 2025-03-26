## Our Deliverables
* [Project Proposal](./proposal%20doc/MARL_Autonomous_Vehicle_Proposal.pdf)
* [Presenting Our Proposal](./MARL_autonomous_vehicle_proposal_presentation.pdf)
* [A Research Summary](./MARL-Lane_Changing-Presentation.pdf)

## Environment Outline
<img src="poc/output_tethered_markers.png" alt="Proof of Concept Output" width="400" height="400"/>

## LaTex Guide for Writing Reports
To work with LaTex locally on your IDE, follow the steps below: 
1. Install [MikTex](https://miktex.org/) as the LaTeX Distribution.
2. Install [Perl](https://strawberryperl.com/).
3. Install the VS Code Extension for previewing the LaTex in a PDF format, such as **LaTeX Workshop**.
4. Previewing and Exporting will be available in the `.tex` file. 

## Prerequisites for Executable Codes in this Project
* Create a Virtual Environment with this command on a terminal: `py -m venv marl_env`.
* Update pip to its latest version with this command: `py -m pip install --upgrade pip`.
* From the Virtual Environment, install all the dependencies with this command: `pip install -r requirements.txt`.
* Or you may find creating a virtual environment useful if you keep facing dependency issues. Just run the following command before installing libraries from pip: 
`
    python -m venv marl_env
    source marl_env/bin/activate  # (On Windowks: marl_env\Scripts\activate)
`.
* Atari is available via Gymnasium: <url>https://www.gymlibrary.dev/environments/atari/index.html</url>
* Clone the repository <url>https://github.com/damat-le/gym-simplegrid.git</url> to get the codes of a simple grid-based environment for customization. 

## Environment's Representation
### Array Representation
The rgb_array shape Image shape: (480, 640, 4) represents the dimensions and color channels of the rendered image of the environment. Here's a breakdown of what each dimension represents:

Breakdown of the Shape
Height (480): The first dimension (480) represents the **height** of the image in pixels.
Width (640): The second dimension (640) represents the **width** of the image in pixels.
Color Channels (4): The third dimension (4) represents the **color channels** of the image. In this case, there are 4 channels, which typically correspond to the RGBA color model:
* R: Red channel
* G: Green channel
* B: Blue channel
* A: Alpha channel (transparency)

Explanation
Height and Width: The height and width of the image determine the resolution of the rendered image. In this case, the image has a resolution of 480x640 pixels.
Color Channels: The 4 color channels (RGBA) provide information about the color and transparency of each pixel in the image. The alpha channel allows for transparency effects, which can be useful for rendering overlapping objects or semi-transparent elements.

### String Representation
When the environment is rendered in ansi mode, the render method generates a string that represents the current state of the environment. This string typically includes information such as the current step, the agent's position, the reward obtained, whether the episode has ended, and the agent's action.

Example:

`Step: 5, Agent Position: (2, 3), Reward: -1, Done: False, Action: (1, 0)`

### Visual Representation for debugging
![alt text](<env_human_render.jpg>)

### Algorithms
#### Leader's Message
[-1, -1, np.float64(2.0), 1, np.float64(1.4142135623730951), 0]