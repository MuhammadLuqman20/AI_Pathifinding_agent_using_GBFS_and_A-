# Dynamic Pathfinding Agent

A Python project implementing **Greedy Best-First Search (GBFS)** and **A*** algorithms with **Manhattan** and **Euclidean** heuristics.  
Supports dynamic obstacles, real-time re-planning, and an interactive **Pygame GUI**.

---

## Features

- **Search algorithms:** A* and GBFS  
- **Heuristics:** Manhattan & Euclidean  
- **Dynamic obstacles:** Obstacles can appear during execution  
- **Interactive grid editor:** Place walls, set start and goal positions  
- **Step-by-step visualization:** Watch the search expand in real-time  
- **Dynamic mode:** Agent re-plans path when new obstacles appear  

---

## Requirements

- Python 3.8+  
- Pygame

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/MuhammadLuqman20/AI_Pathfinding_Agent.git
cd AI_Pathfinding_Agent
```

2. Create a Virtual Environment(Optional but Recommended)

```bash
   python3 -m venv venv
   source venv/bin/activate   # Linux / macOS
   venv\Scripts\activate      # Windows
 ```

3. Install Dependencies
   
```bash
   pip install -r requirements.txt
 ```
---

Running the Code
```bash
python pathfinding_agent.py
```

Controls & Interactions:
Mouse Left Click: Place walls (in edit mode) or move start/goal
Mouse Right Click: Remove walls
Spacebar: Run full search
R key: Reset visualization
G key: Generate random obstacles

Panel options:
Generate/clear map
Choose algorithm: A* or GBFS
Choose heuristic: Manhattan or Euclidean
Select edit mode: Place wall, move start, move goal
Enable dynamic mode
Set grid size & obstacle density
