# Group Assignment 2

# Introduction

This assignment has three problems and answering all three can earn you up to 300 points. It counts toward 10% of your overall grade.

- **Team:** You can work on this assignment in groups of up to three people. You may allocate the problems however you want. I expect teams to collaborate on solving the problems.
- **Deadline:** Week 16, Friday, 23:59

## Corresponding Lecture Notes

These problems require you to review content from the second half of the course, including dynamical systems, optimization-based control, and so on. Refer to the following lecture notes:

[11. Dynamical Systems](https://app.notion.com/p/11-Dynamical-Systems-4ec36cae2e40469a85f5aca077a7d1de?pvs=21) 

[12. State Space Representation](https://app.notion.com/p/12-State-Space-Representation-16ad273a4a0d80519727d9f29d74c44f?pvs=21) 

[13. Numerical Optimization](https://app.notion.com/p/13-Numerical-Optimization-1cad273a4a0d8015a5c1e1dfbc5ee01f?pvs=21) 

[14. Optimization-based Control](https://app.notion.com/p/14-Optimization-based-Control-68083c4d5af24bdfbb048f94022c9fd5?pvs=21) 

[15. Verifying Continuous Interaction](https://app.notion.com/p/15-Verifying-Continuous-Interaction-1cad273a4a0d805ab70ef30a797f1155?pvs=21) 

[16. Bayesian Filtering](https://app.notion.com/p/16-Bayesian-Filtering-10d18eb21c354fb7b8309051de28c869?pvs=21) 

See the following sections for the details of each problem. See the end of the document for deliverables and the grading rubric.

## Deliverables

Submit a **report (PDF)** and your **code**.

- **Report length:** Each problem's report should be around **four A4 pages**, including figures and tables. Keep the total under twelve pages, though you may exceed this if you complete the challenge items at the end of each problem. You may include supplemental material separately if needed. References don't count toward the page limit—add them at the end of the document.
- **Code:** Implement your work in Python. Use the tools we learned in the course when possible, but you may deviate if needed. Your code should be runnable by the evaluators (instructor and TA) and well-organized. Include a link to a repository or submit a zip file. See the boilerplate code in: https://github.com/SMU-HCI/cs702-asg2-spring26

# Problem 1: Dynamical System and Control

## Overview

[flappy.mp4](Group%20Assignment%202/flappy.mp4)

Control theory offers techniques for autonomous agents to regulate movement through input adjustment. In this assignment, you will design a controller that automatically guides a bird's flight in a game inspired by the popular "Flappy Bird." 

In the original game, the player controls a bird that must fly between pipes without colliding with them. In the video above, a small green square represents the bird while moving vertical rectangles represent pipes with gaps. The bird can flap its wings to control vertical acceleration, though it falls due to gravity when not flapping. The bird's horizontal movement is fixed—it cannot be controlled. The bird earns a point and gains a slight horizontal speed boost each time it successfully passes through a gap between the top and bottom sections of a pipe.

Your task is to develop a controller that can guide the bird through the pipes for as long as possible to maximize points. You'll implement a controller that automatically adjusts the bird's vertical acceleration to help it navigate through the gaps without colliding with obstacles.

For the following questions, you must implement a controller and modify the `calculate_control_signal` function using the provided template:

```python
def calculate_control_signal(bird: Bird, pipe: Pipe, controller) -> float:
    """Calculate the control signal for the bird using PID controller.

    Args:
        bird: Current bird state
        pipe: Current pipe state
        pid: PID controller instance

    Returns:
        Control signal value
    """
    # Only consider pipes that are ahead of the bird
    if pipe.x + pipe.w < bird.x:
        return 0

    # Calculate target height (i.e., the middle of the gap)
    target_height = pipe.h + pipe.gap / 2

    # Adjust target height based on bird's velocity
    velocity_offset = bird.vy * 0.2
    adjusted_target = target_height - velocity_offset

    # Get current height and distance to pipe
    current_height = bird.y + bird.h / 2
    distance_to_pipe = pipe.x - (bird.x + bird.w)

    # Adjust control based on distance to pipe
    # More aggressive control when closer to pipe
    # Ensure we don't divide by zero
    if distance_to_pipe <= -1:
        distance_factor = 1.5  # Maximum control factor when very close or past pipe
    else:
        distance_factor = max(0.5, min(1.5, 1 + 1 / (distance_to_pipe + 1)))

    # Calculate and return control signal with bird's velocity for better derivative control
    return controller.calc_input(adjusted_target, current_height, bird.vy) * distance_factor

```

The template script (`game.py`) in the GitHub repo defines the bird's parameters and dynamics. The script includes a `Bird` class that stores the bird's x-y position, speed, and size. The `bird_motion` function controls the dynamics, with gravity set to $-80$. This function updates the bird's vertical position and velocity.

The script also defines the `Pipe` class and `pipe_motion` for the pipe's parameters and behavior. The `Pipe` class stores the position, height, and width of the pipe. The `x` parameter determines the pipe's horizontal position, `h` specifies the height of the bottom pipe section, `gap` defines the vertical space between the bottom and top pipe sections, and `w` sets the pipe's width. The `pipe_motion` function controls the pipe's movement relative to the bird by adjusting its x-coordinate based on the bird's horizontal speed. When a pipe moves beyond the left edge of the window, its x-position resets to the window width, bringing it back to the right side. During this reset, the bottom pipe section's height is randomly adjusted to create a new gap position.

You should not change the parameters and dynamics for the bird and pipe as it would change the game’s difficulty. You should only implement the controller and what’s in `calculate_control_signal`. 

## 1.1. PID controller (45 points)

Recall the PID controller we discussed in the lecture. I’ve provided a placeholder in the [`game.py`](http://game.py) script:

```python
@dataclass
class PIDController:
    Kp: float = 1.5  # Proportional gain
    Ki: float = 0.001  # Integral gain
    Kd: float = 0.5  # Derivative gain
    error_accumulator: float = 0  # Accumulated error for integral term
    prev_error: float = 0  # Previous error for derivative term
    max_accumulator: float = 200  # Anti-windup limit
    dt: float = 1 / 60  # Time step for calculations

    def reset(self) -> None:
        """Reset the controller state."""
        self.error_accumulator = 0
        self.prev_error = 0

    def calc_input(
        self,
        set_point: float,
        process_var: float,
        velocity: float = 0,
        umin: float = -500,
        umax: float = 500,
    ) -> float:
        """Calculate the control signal with improved anti-windup and velocity feedback.

        Args:
            set_point: Target value
            process_var: Current measured value
            velocity: Current velocity (for feedforward control)
            umin: Minimum control output
            umax: Maximum control output

        Returns:
            Control signal value
        """

        # Placeholder for the control signal calculation
        # !!! Implement the PID control algorithm here !!!
        error = set_point - process_var

        return 0
```

#### Tasks

Implement a controller and and use it in the `calculate_control_signal` function to control the bird. Make sure that the bird can fly through gaps of at least 10 pipes. More pipes it can go through, higher the mark.

### Deliverables

Submit the following for this question:

- **Implementation**: Complete, well-documented code implementing both the `PIDController` class and the `calculate_control_signal` function. Include clear comments explaining your implementation approach and reasoning. Include any additional helper functions or utilities you create.
- **Video Demonstration:** Record a screen capture showing the bird's flight behavior under your PID controller. The video should demonstrate how the bird responds to control inputs and navigates through the pipes.
- **Written Report:** Include a detailed explanation of your PID controller implementation, how you tuned the parameters, and any challenges you encountered and your solutions.

## 1.2. Optimization-based controller (45 points)

#### Tasks

Implement a model predictive controller (MPC) by repeatedly optimizing a series of input signals to use in `calculate_control_signal`. Your controller should enable the bird to successfully navigate through gaps at least 10 consecutive pipes. Again, you’ll earn more points if the bird can go through more than 10 pipes.

### Deliverables

Your deliverables for this question are similar to the previous question:

- **Implementation**: Submit your complete code that implements the MPC controller and a revised version of the `calculate_control_signal` function that uses MPC. Include any additional helper functions or utilities you develop (e.g., for path planning).
- **Video Demonstration:** Similar to the previous question, submit the screen capture showing the bird’s flight.
- **Written Report:** Your written report should provide an explanation of how your MPC implementation works. Explain how you formulated the optimization problem mathematically, what prediction and control horizons you selected and why, how you designed the objective function and what constraints you implemented, and so on.
- (Optional) Your could compare MPC to PID performance and examine how the two controllers compare to each other (e.g., in terms of accuracy, computational cost). Discuss key implementation challenges and your solutions.

## 1.3. Human-in-the-Loop Control (10 points)

<aside>
💡

**Note**: You can earn an A without this part. If you score 90/100 points with strong work on Parts 1.1 and 1.2, you’re all set. Only attempt this part if you are interested in applying what you’ve learned to real-world data or are looking for a challenge.

</aside>

In this question, you will design a **human-in-the-loop control system** that assists a human player in controlling the bird. Rather than fully automating the bird's flight, your controller should work collaboratively with human input to improve performance.

The human player provides discrete control inputs (e.g., flapping or not flapping), but may struggle with timing, overreaction, or maintaining optimal trajectory. Your task is to implement a controller that **augments** or **modifies** the human's control inputs to help them achieve better performance.

### Implementation Approaches

You may choose **either PID control or MPC** as the basis for your human-in-the-loop system. Consider the following approaches:

- **Shared Control:** Blend the human's control input with the controller's suggested input. For example, you might weight the human input and autonomous controller output based on factors like distance to obstacles or current error from the optimal trajectory.
- **Input Filtering:** Use the controller to smooth or adjust the human's inputs. For instance, prevent excessive flapping when the bird is already near the target height, or add corrective inputs when the human's actions would lead to collision.
- **Adaptive Assistance:** Adjust the level of assistance dynamically based on performance metrics (e.g., provide more help when the bird is close to obstacles, less help when in safe zones).
- **Predictive Assistance:** Use MPC to predict the outcome of human inputs and modify them if they would lead to failure within the prediction horizon.

You should implement a mechanism(s) to capture human input (keyboard, mouse, or game controller) and integrate it with your chosen control algorithm. The `calculate_control_signal` function should now take into account both the human's intended input and the controller's computed optimal input.

### Deliverables

- **Implementation:** Submit complete code for your human-in-the-loop control system. This should include modifications to capture and process human input, your chosen control algorithm (PID or MPC), and the blending/filtering mechanism that combines human and controller inputs.
- **Video Demonstration:** Record two videos: (1) a human player controlling the bird without assistance, and (2) the same player with your assistance system enabled. The comparison should clearly show the improvement in performance.
- **Written Report:** Your report should include:
    - Description of your control architecture and how human input is integrated
    - Explanation of your blending/filtering strategy and the rationale behind it
    - Quantitative comparison of performance with and without assistance (e.g., average number of pipes cleared, collision rate)
    - Discussion of design trade-offs between autonomy and human agency
    - Reflection on user experience considerations—how does the assistance feel to the human player?
- (Optional) Conduct a small user study with 2-3 participants to evaluate your system. Report on their subjective experience and any observed patterns in how different users interact with the assistance system.

# Problem 2: Trajectory Animation

## Overview

Animated visualizations help analysts understand how objects move through space over time. However, when dozens of trajectories are shown simultaneously, individual paths become tangled and hard to follow. Creating effective animations is therefore an active research topic. Recent work proposes key design considerations for effective animations, such as [Li et al. 2025](https://trajectory-anim.github.io/) (which inspired this problem—I recommend reviewing this research to learn about the context). The design considerations include effective communication of the **global trend** of movement, highlighting **local hotspots** where objects converge or diverge, keeping **occlusion low**, and ensuring motion remains **smooth**. 

In this problem, you will explore how signal temporal logic and gradient-based optimization methods from the course can be applied to information visualization. In Parts 2.1 and 2.2, you will design animations for multiple moving colored dots on a 2D canvas. The starter code generates a synthetic dataset where objects travel from random start positions on the left side of an $800 \times 600$ pixel canvas to random endpoints on the right side. Along the way, objects pass through two hotspots: a **convergence** point (where they gather) near the center-left and a **divergence** point (where they scatter) near the center-right. Part 2.3 extends this work to 3D.

As I mentioned, there is a function `generate_dataset` in the starter code that produces trajectories. Each trajectory is a sequence of $(x, y)$ positions over $K$ time steps. The function accepts the number of trajectories $N$ as a parameter, so you can generate datasets of varying size. For instance, you can run:

```python
from helper import generate_trajectories, export_animation_json

Ps, hotspots = generate_trajectories(5, 60)
export_animation_json(Ps, hotspots, path="trajectories.json")
```

This code generates five trajectories, each with $K = 60$ time steps. The output is saved to `trajectories.json`. To visualize it, run this command from the top-level directory:

```bash
pixi run python problem2/animate.py problem2/trajectories.json 
```

Running this Python script plays an interactive animation like below:

[animation.mp4](Group%20Assignment%202/animation.mp4)

Ultimately, you want to declaratively edit these trajectories using STL and optimization. For instance, you will specify requirements on the timing by which objects enter the convergence hotspot. In the video, notice that the timing by which objects enter the convergence hotspot and leave the divergence hotspot is inconsistent—you will formally specify the bundling so that these objects travel together between the convergence and divergence hotspots.

For the following questions, you will work within the provided code structure in the `problem2/` directory.

## 2.1. STL Specifications for Animation Qualities (30 points)

In this part, you will express desirable animation qualities as **Signal Temporal Logic (STL)** specifications, express them in Python using `stljax`, and evaluate them against trajectory datasets of varying size.

The goal is to formally specify what a good animation should satisfy. You will write STL formulas that capture the following four qualities:

- **Bundling.** Trajectories that share a convergence hotspot should travel close together during the convergence phase. For example, if a group of objects converges at time step $t_c$, the pairwise distances within the group should be small in a time window after $t_c$ until they diverge.
- **Separation.** At every time step, all pairs of objects should maintain a minimum distance so that they remain visually distinguishable (anti-occlusion). Express this as an STL specification.
- **Smoothness.** Object motion should be smooth: the acceleration magnitude should remain bounded over time. Express this as an STL specification. You may define smoothness on a per-object basis as discrete second derivative (acceleration). That is, for each object $i$, compute acceleration at time $t$ as $a^i_t = p^i_{t+1} - 2 \cdot p^i_t + p^i_{t-1}$.
- **Position correctness.** Each object should remain close to specified key points at given times. Specifically, the trajectory should start near its input start position and end near its input end position (within a tolerance), and pass through its respective convergence or divergence hotspots.

### 2.1.1. Two Trajectories

Begin with the simplest case: two trajectories that start at distant positions, meet at a convergence hotspot, travel together briefly, separate at a divergence point, and end at their terminal positions. Using the helper function `generate_trajectories`, creates two random trajectories:

```python
N = 2  # number of trajectories
steps = 10  # number of time steps
Ps, hotspots = generate_trajectories(N, steps)
```

Consider a concrete example: let’s say we have two trajectories $P^0$ and $P^1$ that ranges from $t \in \{0 \dots 9 \}$. So, $P^0 = \{p^0_0, p^0_1, \dots p^0_9\}$ and $P^1 = \{p^1_0, p^1_1, \dots p^1_9\}$, where each point $p^i_t$ represents the 2D position $(x, y)$ of trajectory $i$ at time step $t$. And these trajectories should satisfy the folloiwng:

1. At $t=0$: $p^0_0$ and $p^1_0$ are at distant starting positions with the error at most the threshold $\epsilon_{start}$.
2. At $t=2$: Both trajectories meet at the convergence hotspot with the error at most the threshold $\epsilon_{conv}$.
3. For $t \in [2, 7]$, the trajectories travel together with the minimum distance $\delta_{min}$
4. At $t=7$: They pass through the divergence hotspot with error at most $\epsilon_{div}$ and separate.
5. At $t=9$: $p^0_9$ and $p^1_9$ are at their respective terminal positions with the error at most the threshold $\epsilon_{end}$.

#### Tasks

**STL Representation.** Write the formal STL formula for each of the five specifications listed above. Express each constraint using standard STL temporal operators (e.g., $G$ for "always", $F$ for "eventually", $U$ for "until") and predicates involving distance, position, and thresholds. Use interval notation to specify time bounds where applicable. For example, $G_{[t_1, t_2]} \phi$ means "property $\phi$ holds at all times in the interval $[t_1, t_2]$."

**Implementation.** Translate each STL formula from Task 1 into executable code using the `stljax` library. 

- Construct temporal operators using `Always` and `Eventually` from `stljax.formula`. Specify the correct time intervals for each operator when appropriate.
- Test your implementation with the example trajectories generated by `generate_trajectories(2, 10)`. Print the robustness value and verify that it behaves as expected (e.g., trajectories that violate the constraints should produce negative robustness).

Include comments in your code explaining which part of the STL formula each code block implements.

### 2.1.2. Multiple 2D Trajectories

Now generalize the problem to handle multiple 2D trajectories using the `generate_trajectories` helper function. 

```python
N = 3
steps = 10
Ps, hotspots = generate_trajectories(N, steps)
```

Using the function above, generate $\{3, 5, 10\}$ trajectories. Like the problem in Part 2.1.1, each trajectory $P^i$ starts at a different position, meets at the convergence hotspot at $t = t_{conv}$, travels together between the hotspots, separates at the divergence hotspot at $t = t_{div} (> t_{conv})$, and ends at its respective terminal position.

#### Tasks

**STL Representation.** As in Part 2.1.1, represent the specification using STL. Describe in detail where the specification differs from Part 2.1.1.

**Implementation.** Implement the STL specifications using `stljax.`

### Deliverables

For Part 2.1.1 and 2.1.2, submit the following:

- **Written Report**: Include:
    - The formal STL formula for each specification in mathematical notation
    - A short description of how each formula is implemented using `stljax`
    - A table or figure showing robustness values for each case with different numbers of trajectories.
    - Discussion the values of the robustness and report insights.
- **Implementation**: Complete code implementing the STL specifications. Include clear comments explaining each STL formula and how it maps to the animation quality.

## 2.2. Optimizing Animations (30 points)

Using the STL specifications from Part 2.1, optimize the trajectory positions so that the animation satisfies all four specifications simultaneously as much as possible. Rather than writing procedural animation logic, you will use the STL robustness as a differentiable loss function and solve for positions via gradient descent.

### 2.2.1. Optimizing Two Trajectories

The decision variable is a JAX array `pos` with shape `(K, N, 2)`. The optimization loop should use non-linear optimizer (e.g., gradient-based method) to minimize a combined loss.

### Tasks

Generate two trajectories. Your goal is to implement the loss function which you can use in optimization loop to improve the trajectories. You must implement the following:

**Implement loss functions.** Use the STL specifications and implementation from Part 2.1 to define a loss function for each specification: `bundling_loss`**,** `separation_loss`, `smoothness_loss`, and `position_loss`. Combine the output of these loss functions to calculate the total loss, reflecting bundling, separation, smoothness, and positional correctness. The `total_loss` function should take a weighted average of the individual losses. You may add other loss terms if appropriate.

**Run the optimization and analyze results.** Optimize trajectories for $N = 2$ and produce an animation. Then perform two analyses:

- **Comparison**: Compare the original, unmodified trajectories with the optimized animation. Plot them visually as static figures and play them as animations. Discuss the effect of optimization on the trajectories.
- **Sensitivity analysis**: Adjust the weight parameters to change which losses have higher priority. For example, reduce the bundling loss weight and observe how the optimized trajectories change.

### 2.2.2. Optimizing Multiple Trajectories

### Tasks

Now, you repeat Part 2.2.2. But instead of using two trajectories, create ten trajectories and perform trajectory optimization.

- **Implement loss functions.** Implement the loss functions by integrating STL specifications. If your code from Part 2.2.1 already supports handling multiple trajectories, you don’t have to do anything here.
- **Run the optimization and analyze results.** Optimize trajectories for $N = 10$ and produce an animation.

Perform the same analyses as Part 2.2.1:

- **Comparison**: Compare the original, unmodified trajectories and the optimized animation. Plot them visualy as static figures as well as play as animations. Discuss the effect of optimization on the trajectories.
- **Sensitivity analysis**: Adjust the weight parameters to change which losses have higher priority. For example, reduce the weight for the bundling loss and observe how the optimized trajectories change.

### Deliverables (Implementation, Video, and Written Report)

Submit the following for this question:

- **Implementation**: Complete code for the loss functions integrating STL specifications from Part 2.1 and the optimization loop. Include clear comments explaining your approach.
- **Video Demonstration**: Record a screen capture showing the unoptimized and optimized animation.
- **Written Report**: Your report should include:
    - The loss function formulation and weight choices
    - Convergence evidence (loss curves over optimization steps)
    - Quantitative comparison of baseline vs. optimized animation using evaluation metrics
    - An sensitivity evaluation showing the effect of adjusting weigtage.

## 2.3. 3D Trajectory Optimization (30 points)

In this part, you will extend the STL-based robustness calculation and trajectory optimization from Parts 2.1 and 2.2 to **three-dimensional** space. The shift from 2D to 3D is conceptually and programmatically incremental but introduces new challenges, as pairwise distances, convergence, and separation all operate in $\mathbb{R}^3$.

### 3D Dataset

The starter code in `part2_2.py` provides a `generate_dataset_3d` function that produces synthetic 3D trajectories in a $800 \times 600 \times 400$ volume. Objects start on the left face ($x \approx 0$), pass through convergence and divergence hotspots in the interior, and exit on the right face ($x \approx 800$). Each trajectory is a sequence of $(x, y, z)$ positions over $K = 60$ time steps.

```python
traj_in_3d, hotspots_3d = generate_dataset_3d(n=20)  # shape (N, K, 3)
```

The `Hotspot3D` dataclass extends the 2D `Hotspot` with a $z$-coordinate:

```python
@dataclass
class Hotspot3D:
    x: float
    y: float
    z: float
    kind: str          # "converge" | "diverge"
    group: list[int]
    time_step: int
```

### Tasks

**Adapt STL specifications to 3D.** Extend each of the four STL robustness functions (bundling, separation, smoothness, start/end correctness) to work with 3D positions of shape `(K, N, 3)`. The core logic should be analogous to the 2D versions, but all distance and acceleration computations must operate in $\mathbb{R}^3$.

**Implement loss functions:** Following Part 2.2, implement loss functions that you will use for optimization in the next step.

**Optimize 3D trajectories.** Extend the optimization pipeline from Part 2.2 to work in 3D. The decision variable is now of shape `(K, N, 3)`. Use the same Adam optimizer approach, but with the 3D STL robustness and loss functions. (To create a smooth animation, you would need to play around with the weights for each loss).

**Visualize.** Create visualizations of your results. Your visualization should include: (i) 3D points representing each object's position at each time step, and (ii) convergence and divergence hotspots displayed as 3D spheres or markers with appropriate colors (e.g., red for convergence, blue for divergence). Since `matplotlib` and `pygame` from earlier parts aren't well-suited for 3D data, consider using the [Rerun SDK](https://rerun.io/docs) to build an interactive 3D visualization.

### Deliverables

- **Written Report**: Discuss
    - A discussion of how each STL specification adapts—or remains unchanged—when transitioning from 2D to 3D
    - Comparison of baseline vs. optimized 3D trajectories
    - Screenshots of the 3D visualization from multiple viewpoints, showing the convergence and divergence structure and the separation between trajectories
    - Discussion of any new challenges introduced by the third dimension
- **Implementation**: Implement:
    - Translate the STL specifications into `stljax`.
    - Implement loss functions and 3D trajectory optimization pipeline
    - Visualization code (e.g., using Rerun).

## 2.4. Animations with Multiple Hotspots (10 points)

<aside>
💡

**Note**: You can earn an A without this part. If you score 90/100 points with strong work on Parts 2.1, 2.2, and 2.3, you're all set. Only attempt this part if you are interested in applying what you've learned to research or looking for a challenge.

</aside>

The paper that inspired this assignment problem handles animations with more than two hotspots ([Li et al. 2025](https://trajectory-anim.github.io/)). In this part, you will build an interactive **authoring tool** that lets a user specify animation through a graphical interface. The authoring tool should lets the user do:

- **Hotspot authoring**: Click to place a hotspot on the canvas, choose whether it is a convergence or divergence point, and select which group of objects it affects.

Then, the tool should also implement some of the following features:

- **Constraint weight sliders**: Adjust the relative weights of the loss terms (bundling vs. separation vs. smoothness vs. deviation) and re-run the optimization to see how the animation changes.
- **Playback + timeline scrub**: A timeline bar that allows scrubbing through the animation, with visual indicators showing where hotspots are located in time.
- **Group definition UI**: Lasso or click-to-select objects to define custom groups for bundling or hotspot assignment.
- **Trajectory count control**: A control to adjust the number of trajectories $N$ and regenerate the dataset, allowing users to explore how the animation behaves at different scales.
- **STL specification editor**: A panel that displays the active STL specifications and allows users to adjust parameters (e.g., bundling threshold, separation distance, smoothness bound) and see updated robustness values in real time.

You can implement these features programmatically by editing your scripts from Parts 2.1–2.3. However, the goal is to provide an interactive interface for designing animations—not require users to edit code. You can use any GUI frameworks to build this tool.

After building the GUI tool, conduct a small informal user study. Recruit **3–5 participants** and assign them a simple task to test usability. Collect metrics such as task completion time and responses to an open-ended question about their experience.

### Deliverables

Submit the following for this question:

- **Implementation**: Submit complete code for your authoring tool, including the UI components and integration with the Part 2.2 optimization pipeline.
- **Video Demonstration**: Record a screen capture showing a user interacting with the authoring tool to create an animation. The video should demonstrate at least three of the features listed above and show the resulting optimized animation.
- **Written Report**: Your report should include:
    - Description of the authoring tool's interface and features
    - Discussion of design decisions
    - Screenshots or diagrams of the interface
    - Results from the mini user study
    - Discussion of actionable insights from the study

# Problem 3: Arm Impedance Model

## Overview

The human arm exhibits remarkable control during reaching movements — smoothly guiding the hand from a resting position to a target while adapting to perturbations. Understanding the mechanics behind this behavior is central to fields such as rehabilitation robotics, prosthetics design, and human-computer interaction.

In this problem, you will model the human arm’s endpoint behavior as a **linear mass-spring-damper system** (the endpoint impedance model introduced by Hogan, 1985). The governing equation is:

$$
M\,\ddot{p} + B\,\dot{p} + K\,(p - p_{eq}) = F_{ext}
$$

where $p$ is the hand position, $p_{eq}$ is the equilibrium position, $M$ is the effective mass matrix, $B$ is the damping matrix, $K$ is the stiffness matrix, and $F_{ext}$ is any external force applied to the hand. The **equilibrium point hypothesis** proposes that the central nervous system controls reaching by shifting $p_{eq}$ from the current hand position to the target. The impedance dynamics then govern how the hand moves to the new equilibrium.

You will progressively build this model from 1D to 3D, simulate reaching movements, apply state estimation techniques, and finally connect the model to real motion data captured via MediaPipe.

**Notation convention.** Throughout this problem, the impedance model parameters are denoted $M$ (mass), $B$ (damping), and $K$ (stiffness). When writing the state space form $\dot{\mathbf{x}} = A_c\,\mathbf{x} + B_c\,u$, we use subscript $c$ (continuous) on the state space input matrix to distinguish it from the damping matrix $B$. Similarly, discrete-time state space matrices are written $A_d$, $B_d$, etc.

The starter code (`problem3/starter.py`) in the GitHub repository provides the `build_state_space`, `min_jerk_profile`, and `simulate_smooth_reaching` functions described in the lecture notes. You should use these as building blocks throughout this problem.

## 3.1. One-Dimensional State Space Form (20 points)

Begin with the simplest case: a 1D mass-spring-damper model of the arm endpoint moving along a single axis. The equation of motion is:

$$
m\,\ddot{x} + b\,\dot{x} + k\,(x - x_{eq}) = F_{ext}
$$

Use the following parameters (representative of lateral hand movement):

```python
m = 1.5    # kg (effective mass)
b = 15.0   # N·s/m (damping)
k = 200.0  # N/m (stiffness)
```

### Tasks

**Derive the state space form.** Define the state vector $\mathbf{x} = [\delta x,\ \dot{x}]^\top$ where $\delta x = x - x_{eq}$, and express the system in standard state space form:

$$
\dot{\mathbf{x}} = A_c\,\mathbf{x} + B_c\,u, \qquad y = C\,\mathbf{x}
$$

where $u = F_{ext}$. Write out $A_c$ and $B_c$ symbolically (using letters like $m$ and $k$) and numerically with the given parameter values. Assume we can only observe the hand displacement $\delta x$, not the speed $\dot{x}$. Derive $C$.

**Simulate a step response.** The hand starts at $x_{start} = 0.30$ m and the equilibrium is instantaneously shifted to $x_{target} = 0.45$ m (i.e., $\delta x(0) = -0.15$ m, $\dot{x}(0) = 0$). Assume no external force ($F_{ext} = 0$). Simulate this step response for 2 seconds using [`scipy.signal.lsim`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lsim.html). Plot (i) Hand position $x(t)$ vs. time and (ii) Hand velocity $\dot{x}(t)$ vs. time. Describe the physical implications of the resulsts that you observe.

**Parameter exploration.** Vary the damping coefficient $b$ across three values (e.g., $b \in \{5, 15, 40\}$) while keeping $m$ and $k$ fixed. Simulate the step response for each case. Discuss how damping ratio affects overshoot, settling time, and oscillation.

### Deliverables

- **Implementation**: Complete code that builds the 1D state space model, simulates the step response, and produces the required plots.
- **Written Report**: Include the symbolic derivation of $A_c$, $B_c$, $C$ and the parameter exploration plots with discussion.

## 3.2. Two-Dimensional State Space Form (40 points)

### 3.2.1. Derivation and Simulation

Extend the model to 2D, representing the hand moving in the horizontal plane (e.g., reaching across a table viewed from above). The equation of motion becomes:

$$
M\,\ddot{p} + B\,\dot{p} + K\,(p - p_{eq}) = F_{ext}
$$

where $p = [x, y]^\intercal$ and $M$, $B$, $K$ are $2 \times 2$ symmetric positive definite matrices. Use the following parameters:

```python
import numpy as np

M_2d = np.array([[1.5, 0.3], [0.3, 2.0]])
B_2d = np.array([[15.0, 3.0], [3.0, 20.0]])
K_2d = np.array([[200.0, 50.0], [50.0, 350.0]])
```

### Tasks

**Derive the state space matrices.** Define the state vector $\mathbf{x} = [\delta x ,\  \delta y ,\ \dot{x} ,\ \dot{y}]^\intercal = [\delta p^\intercal,\ \dot{p}^\intercal]^\intercal \in \mathbb{R}^4$ and derive the matrices $A_c$, $B_c$, and $C$ symbolically. Assume we can observe displacement $\delta p$ but not velocity $\dot{p}$ when deriving $C$. Compute them numerically using the given parameters. 

**Simulate a 2D reach.** The hand starts at $p_{start} = [0.30, 0.20]^\intercal$ and reaches for a cup at $p_{target} = [0.45, 0.35]^\intercal$ using a step change in $p_{eq}$. Simulate for 2 second and plot:

- Hand trajectory in the $x$$y$ plane.
- Position components vs. time
- Speed profile $\|\dot{p}(t)\|$ vs. time

### 3.2.2 Smooth Reaching with a Minimum-Jerk Profile

The step-change model from Part 3.2.1 produces overshoot and oscillation, which is unrealistic. A more biologically plausible model shifts $p_{eq}(t)$ smoothly from start to target using a **minimum-jerk trajectory profile** (Flash & Hogan, 1985):

$$
s(t) = 10\left(\frac{t}{T}\right)^3 - 15\left(\frac{t}{T}\right)^4 + 6\left(\frac{t}{T}\right)^5
$$

where $s(t)$ goes from 0 to 1 over the reach duration $T$, with zero velocity and acceleration at both endpoints.

The equilibrium then shifts as:

$$
p_{eq}(t) = p_{start} + s(t) \cdot (p_{target} - p_{start})
$$

### Tasks

**Implement smooth reaching.** See the file `problem3/part_3_2_2.py` in the assignment repository. Edit the function `simulate_smooth_reaching` by editing the parts annotated with `TODO`. 

Then, use the edited function to simulate the 2D reach from Part 3.2.1 with $T_{reach} = 0.5$ s and $T_{total} = 1.0$ s. Plot:

- Hand trajectory in the $x$$y$ plane
- Position components vs. time
- Speed profile

Compare the step-change responses from Part 3.2.1 with the minimum-jerk responses. Discuss which model produces a more plausible representation of actual human reaching.

**Reach duration study.** Vary the reach duration $T_{reach} \in \{0.3, 0.5, 0.8, 1.2\}$ with $T_{total} = T_{reach} + 0.5$ and plot the resulting speed profiles. Discuss how reach duration affects the hand trajectory and velocity profile shape.

### Deliverables

- **Implementation**: Complete code for both Part 3.2.1 and Part 3.2.2, including all simulations and plots.
- **Written Report**: Include the symbolic derivation of $A_c$, $B_c$, $C$, $D$ for the 2D system, comparison between step-change and minimum-jerk responses, and discussions as described above. Use figures to support your analysis.

## 3.3. Three-Dimensional State Space Form (30 points)

Extend the 2D model to 3D to represent general arm movements, such as reaching up to a cup on a shelf or pointing at a screen. The $M$, $B$, and $K$ matrices become $3 \times 3$, and the state vector expands to $\mathbb{R}^6$. Use the following parameters:

```python
import numpy as np

M_3d = np.array([[1.5, 0.3, 0.0], [0.3, 2.0, 0.0], [0.0, 0.0, 1.0]])
B_3d = np.array([[15.0, 4.0, 0.0], [4.0, 25.0, 0.0], [0.0, 0.0, 10.0]])
K_3d = np.array([[200.0, 60.0, 0.0], [60.0, 400.0, 0.0], [0.0, 0.0, 100.0]])
```

### Tasks

**3D reaching simulation.** Simulate a reach from $p_{start} = [0.25, 0.20, 0.10]^\intercal$ to $p_{target} = [0.40, 0.40, 0.30]^\intercal$ using the minimum-jerk profile with $T_{reach} = 0.5$ s and $T_{total} = 1.0$ s. You should be able to use the code from Part 3.2.2. Produce a 3D trajectory plot (using `mpl_toolkits.mplot3d` or similar 3D visualization tool). 

### Deliverables (Implementation and Written Report)

- **Implementation**: Complete code for the task and 3D plot production.
- **Written Report**: Include the plot and discussion.

## 3.4. Arm Tracking and System Identification (10 points)

<aside>
💡

**Note**: You can earn an A without this part. If you score 90/100 points with strong work on Parts 3.1–3.3, you’re all set. Only attempt this part if you are interested in applying what you’ve learned to real-world data or are looking for a challenge.

</aside>

In this part, you will record real arm movements using a webcam, extract wrist trajectories via [MediaPipe Pose](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker), and use the collected data for system identification. This connects the theoretical model you have built to actual human motion.

### Data Collection

Record **at least 5 short videos** (2–3 seconds each) of reaching movements using your webcam or phone:

1. Sit at a table with your right arm visible to the camera
2. Start with your hand at a resting position on the table
3. Reach smoothly to a target object (e.g., a cup, phone, or marked spot) approximately 20–30 cm away
4. Hold the final position briefly
5. Keep your torso still to isolate arm movement

Vary the target position across recordings (e.g., reach forward, laterally, diagonally) to capture different aspects of the arm’s impedance.

### Tasks

**Trajectory extraction.** Write a `extract_wrist_trajectory` function that uses MediaPipe Pose to extract 3D wrist trajectories from each video. The function normalizes coordinates to approximate metric units using the shoulder-to-elbow distance as a reference. For each video, plot the raw extracted trajectory (position vs. time for all three axes).

**State estimation.** Apply filtering or smooth method to the extracted trajectories to obtain smoothed position and estimated velocity signals (e.g., MHE, Kalman filter). Compare the filtered output with the raw data to demonstrate the benefit of filtering.

**System identification.** Using the filtered trajectories from all videos, estimate the arm's impedance parameters ($B$ and $K$, assuming $M$ is known from the literature). Fit on all videos jointly as in Part 3.4(B). Report: (i) estimated parameters, (ii) prediction RMSE for each video, and (iii) a figure comparing fitted model trajectories with observed data for at least one video.

**Discussion.** Reflect on the results. Discuss questions such as: How plausible are the estimated parameters compared to common sense or typical values reported in the literature? What are the main sources of error in the pipeline? Is the impedance-based model fundamentally a good model for describing human arm motion?

**Go Beyond.** Explore better ways to model human arm motion for interactions like reaching an item on a shelf and touching a display.

### Deliverables (Implementation, Videos, and Written Report)

- **Implementation**: Complete code for trajectory extraction, state estimation on real data, and system identification.
- **Videos**: Submit at least 5 recorded reaching videos used for data collection, along with a demo video showing the MediaPipe keypoint overlay on at least one recording.
- **Written Report**: Include at least:
    - Description of your data collection setup and procedure
    - Raw trajectory plots for each video and filtered results on real data.
    - System identification results
    - Discussion of estimated parameters, error sources, etc.

# References