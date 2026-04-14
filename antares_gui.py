import py5
import subprocess
import threading
import sys

STATE_WELCOME = 0
STATE_RESTING_RECORDING = 2
STATE_NEUROFEEDBACK_INSTRUCTIONS = 3
STATE_NF_RECORDING = 4
STATE_END = 5

current_state = STATE_WELCOME
script_running = False
subject_id = None
age = None
sex = None
visit = None

NEUROFEEDBACK_INSTRUCTIONS_TEXT = (
    "In this session, you will see a tree on the screen.\n\n"
    "Your goal is to maintain a calm, relaxed focus.\n"
    "When you succeed in relaxing, you might notice the tree begin to grow.\n\n"
    "Simply observe the tree, and let your internal state guide its changes."
)

def setup():
    global subject_id, age, sex, visit
    
    # py5.full_screen(py5.P2D) 
    py5.size(800, 600)
    py5.smooth(8) 
    py5.no_cursor()
    
    # Now get user data from command line arguments
    if len(sys.argv) == 5:
        subject_id = sys.argv[1]
        age = sys.argv[2]
        sex = sys.argv[3]
        visit = sys.argv[4]
        print(f"GUI started with data: {subject_id}, {age}, {sex}, {visit}")
    else:
        print("Error: GUI requires 4 arguments (subject_id, age, sex, visit)")
        print(f"Received {len(sys.argv)-1} arguments: {sys.argv[1:]}")
        py5.exit_sketch()

def draw():
    """ The core experiment loop, called every frame. """
    py5.background(0)
    
    if current_state == STATE_WELCOME:
        draw_welcome_screen() 
        
    elif current_state == STATE_RESTING_RECORDING:
        draw_resting_recording_stage()
        
    elif current_state == STATE_NEUROFEEDBACK_INSTRUCTIONS:
        draw_neurofeedback_instructions_screen()

    elif current_state == STATE_NF_RECORDING:
        draw_nf_screen()
        
    elif current_state == STATE_END:
        draw_end_screen()

def draw_text_prompt_at_bottom():
    """ Standard prompt to advance the stage. """
    italic_font = py5.create_font("Helvetica-Oblique", 20)
    py5.text_font(italic_font)
    py5.text("Press the space bar to continue.", py5.width / 2, py5.height * 0.85)

def draw_welcome_screen():
    title_font = py5.create_font("Helvetica-Bold", 32)
    body_font = py5.create_font("Helvetica", 24)
    italic_font = py5.create_font("Helvetica-Oblique", 20)

    py5.fill(255)
    py5.text_align(py5.CENTER, py5.CENTER)
    
    py5.text_font(title_font)
    py5.text("Welcome to ANTARES", py5.width / 2, py5.height * 0.18)
    
    py5.text_font(italic_font)
    py5.text("Advancing Neurofeedback in Tinnitus", py5.width / 2, py5.height * 0.25)
    
    py5.text_font(body_font)
    py5.text_leading(35)
    instr = (
        "In the first stage, we will conduct a short resting-state recording.\n"
        "During this time, please keep your gaze fixed on the cross."
    )
    py5.text(instr, py5.width / 2, py5.height / 2)
    draw_text_prompt_at_bottom()

def draw_resting_recording_stage():
    py5.stroke(255)
    py5.stroke_weight(2)
    py5.line(py5.width/2 - 45, py5.height/2, py5.width/2 + 45, py5.height/2)
    py5.line(py5.width/2, py5.height/2 - 45, py5.width/2, py5.height/2 + 45)

def draw_neurofeedback_instructions_screen():
    title_font = py5.create_font("Helvetica-Bold", 32)
    body_font = py5.create_font("Helvetica", 24)
    italic_font = py5.create_font("Helvetica-Oblique", 20)

    py5.fill(255)
    py5.text_align(py5.CENTER, py5.CENTER)
    py5.text_font(title_font)
    py5.text("You finished the resting-state section", py5.width / 2, py5.height * 0.18)
    
    py5.text_font(italic_font)
    py5.text("Now we will start the main neurofeedback session.", py5.width / 2, py5.height * 0.25)
    py5.text_font(body_font)
    py5.text_leading(40) 
    py5.text(NEUROFEEDBACK_INSTRUCTIONS_TEXT, py5.width / 2, py5.height * 0.55)
    draw_text_prompt_at_bottom()

def draw_nf_screen(): # wait for Delphine
    title_font = py5.create_font("Helvetica-Bold", 32)
    body_font = py5.create_font("Helvetica", 24)
    italic_font = py5.create_font("Helvetica-Oblique", 20)

    py5.fill(255)
    py5.text_align(py5.CENTER, py5.CENTER)
    py5.text_font(title_font)
    py5.text("Here some nice visualization from ECAL Lab", py5.width / 2, py5.height * 0.18)
    
    py5.text_font(italic_font)
    py5.text("waiting for Delphine to provide the last version of the viz ...", py5.width / 2, py5.height * 0.25)
    py5.text_font(body_font)
    # py5.text_leading(40) 
    # py5.text(NEUROFEEDBACK_INSTRUCTIONS_TEXT, py5.width / 2, py5.height * 0.55)
    # draw_text_prompt_at_bottom()

def draw_end_screen():
    title_font = py5.create_font("Helvetica-Bold", 32)
    italic_font = py5.create_font("Helvetica-Oblique", 20)
    
    py5.fill(255)
    py5.text_align(py5.CENTER, py5.CENTER)
    py5.text_font(title_font)
    py5.text("Session Completed.", py5.width / 2, py5.height / 2)
    
    py5.text_font(italic_font)
    py5.text("Thank you for your participation", py5.width / 2, py5.height * 0.85)

def run_script_async(script_name, args, callback):
    """Run a Python script in a background thread and call callback when done."""
    global script_running
    script_running = True
    
    def run():
        global script_running
        try:
            cmd = ["python", script_name] + args
            print(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"Completed: {script_name}")
            if result.stdout:
                print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"Error running {script_name}: {e}")
            if e.stderr:
                print(e.stderr)
        finally:
            script_running = False
            callback()
    
    thread = threading.Thread(target=run, daemon=True)
    thread.start()

def transition_to_resting():
    """Callback after antares_2.py finishes"""
    global current_state
    print("Transitioning to resting recording stage")
    current_state = STATE_RESTING_RECORDING
    run_script_3()

def run_script_3():
    """Run antares_3.py and antares_4.py, and when it finishes, transition to neurofeedback instructions"""
    global subject_id, age, sex, visit
    
    def after_script_3():
        global current_state
        print("Transitioning to neurofeedback instructions")
        current_state = STATE_NEUROFEEDBACK_INSTRUCTIONS
    
    args = ["--subject_id", subject_id, "--age", str(age), "--sex", sex, "--visit", str(visit)]
    run_script_async("antares_3.py", args, after_script_3)

def transition_to_end():
    """Callback after antares_5.py finishes"""
    global current_state
    print("Transitioning to end screen")
    current_state = STATE_END

def key_pressed():
    """ Controls the progression of the experiment state. """
    global current_state, script_running, subject_id, age, sex, visit
    
    print(f'Key pressed: {py5.key}, Current state: {current_state}')
    
    if py5.key == ' ' and not script_running:
        print(f'Space bar pressed in state {current_state}')
        
        if current_state == STATE_WELCOME:
            current_state = STATE_RESTING_RECORDING
            print("Starting antares_2.py...")
            args = ["--subject_id", subject_id, "--visit", visit]
            run_script_async("antares_2.py", args, transition_to_resting)

        
        elif current_state == STATE_NEUROFEEDBACK_INSTRUCTIONS:
            current_state = STATE_NF_RECORDING
            print("Starting antares_5.py...")
            # Start antares_5.py
            args = ["--subject_id", subject_id, "--visit", visit]
            run_script_async("antares_5.py", args, transition_to_end)
    
    # Safety: Esc to exit full screen mode
    if py5.key == py5.ESC:
        py5.exit_sketch()

if __name__ == "__main__":
    py5.run_sketch()