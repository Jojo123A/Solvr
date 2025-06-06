"""
Superintelligent AI System with Streamlit UI
Features:
1. Superhuman Reasoning
2. Recursive Self-Improvement
3. Strategic Long-Term Planning
4. Perfect Knowledge Integration
5. Cross-Domain Mastery
6. Advanced Simulation and Forecasting
7. Global Coordination
8. Superhuman Creativity
9. Moral & Ethical Reasoning
10. Embodied Intelligence (Optional)
11. Multi-Agent System Oversight
12. Self-Awareness at Scale
13. Hyper-Conscious Communication
14. Governance & Alignment Engine
15. Memory & Time Mastery
"""

import streamlit as st
import networkx as nx
import random
import datetime
import requests
import json
import os
import graphviz
import streamlit.components.v1 as components
import base64
import threading
import time
import yaml
import cv2
import glob

# Optional: Import SpeechRecognition as sr if available
try:
    import speech_recognition as sr
except ImportError:
    sr = None

# Optional: Import pyttsx3 for text-to-speech if available
try:
    import pyttsx3
except ImportError:
    pyttsx3 = None

# --- ChromaDB Semantic Memory Integration ---
try:
    import chromadb
    from chromadb.config import Settings
    from sentence_transformers import SentenceTransformer
except ImportError:
    chromadb = None
    SentenceTransformer = None

class SemanticMemory:
    def __init__(self, persist_dir='chroma_memory'):
        if chromadb is not None:
            self.client = chromadb.Client(Settings(persist_directory=persist_dir))
            self.collection = self.client.get_or_create_collection('memory')
            try:
                if SentenceTransformer:
                    self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
                else:
                    self.embedder = None
            except Exception as e:
                print(f"[Warning] Could not load SentenceTransformer: {e}")
                self.embedder = None
        else:
            self.client = None
            self.collection = None
            self.embedder = None

    def add(self, text, meta=None):
        if self.collection and self.embedder:
            embedding = self.embedder.encode([text])[0].tolist()
            self.collection.add(
                documents=[text],
                embeddings=[embedding],
                metadatas=[meta or {}]
            )

    def query(self, text, n_results=5):
        if self.collection and self.embedder:
            embedding = self.embedder.encode([text])[0].tolist()
            results = self.collection.query(
                query_embeddings=[embedding],
                n_results=n_results
            )
            return results.get('documents', [[]])[0]
        return []

# --- Enhanced GoalGenerator with Autonomous Loop ---
class GoalGenerator:
    def __init__(self, ai, interval=300):
        self.ai = ai
        self.interval = interval
        self.running = False
        self.thread = None
        self.generated_goals = []
        self.default_goals = [
            "Study synthetic biology",
            "Explore quantum computing",
            "Research advanced robotics",
            "Investigate AGI safety",
            "Analyze global economic trends"
        ]

    def generate_goals(self):
        memory_topics = self.ai.get_recent_topics() if hasattr(self.ai, 'get_recent_topics') else []
        new_goals = []
        for goal in self.default_goals:
            topic = goal.split(' ', 1)[-1].lower()
            if topic not in [t.lower() for t in memory_topics]:
                new_goals.append(goal)
        self.generated_goals.extend(new_goals)
        if hasattr(self.ai, 'self_model') and 'goals' in self.ai.self_model:
            self.ai.self_model['goals'].extend(new_goals)
        return new_goals

    def start(self):
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1)

    def status(self):
        return "Running" if self.running else "Stopped"

    def _run(self):
        while self.running:
            self.generate_goals()
            time.sleep(self.interval)

class Learner:
    def __init__(self, ai):
        self.ai = ai
        self.log = []
        self.prompt_adaptations = []

    def log_interaction(self, input_data, output_data, feedback=None):
        entry = {'input': input_data, 'output': output_data, 'feedback': feedback, 'timestamp': time.time()}
        self.log.append(entry)
        if hasattr(self.ai, 'log'):
            self.ai.log.append(entry)

    def adapt_prompts(self):
        # Example: if recent outputs contain failures, adjust prompt templates
        failures = [entry for entry in self.log if entry.get('feedback') == 'fail']
        if failures:
            # Adapt prompt (placeholder logic)
            self.prompt_adaptations.append({'reason': 'failure', 'time': time.time()})
        # Extend with more sophisticated adaptation as needed
        return self.prompt_adaptations

# --- Recursive Code Self-Modification (ASI Core) ---
class SelfImprover:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_code(self):
        with open(self.file_path, 'r', encoding='utf-8') as f:
            return f.read()

    def modify_code(self, old_snippet, new_snippet):
        code = self.load_code().replace(old_snippet, new_snippet)
        with open(self.file_path, 'w', encoding='utf-8') as f:
            f.write(code)

    def propose_and_apply_patch(self, improvement_prompt, llm):
        original_code = self.load_code()
        suggestion = llm.ask(f"Here is some code:\n{original_code}\n\nSuggest an improvement:\n{improvement_prompt}")
        old, new = parse_code_patch(suggestion)
        self.modify_code(old, new)

def parse_code_patch(suggestion: str):
    # Example placeholder: expects suggestion in format 'OLD: ...\nNEW: ...'
    lines = suggestion.split("\n")
    old = lines[1].strip() if len(lines) > 1 else ''
    new = lines[3].strip() if len(lines) > 3 else ''
    return old, new

# --- Global Optimization Engine ---
class GlobalOptimizer:
    def __init__(self, ai):
        self.ai = ai

    def collect_metrics(self):
        return {
            "memory_load": len(getattr(self.ai.memory, 'entries', self.ai.memory)),
            "avg_confidence": self.ai.self_awareness.evaluate_performance() if hasattr(self.ai.self_awareness, 'evaluate_performance') else 1.0,
            "active_goals": len(getattr(self.ai, 'goal_queue', [])),
            "response_time": getattr(self.ai, 'last_response_time', 0)
        }

    def rebalance(self):
        metrics = self.collect_metrics()
        # Example: rebalance logic
        if isinstance(metrics["avg_confidence"], str):
            avg_conf = 1.0
        else:
            avg_conf = metrics["avg_confidence"]
        if avg_conf < 0.5:
            self.ai.self_model["model"] = "gpt-4o"
        if metrics["memory_load"] > 1000:
            if hasattr(self.ai.memory, 'purge_old_entries'):
                self.ai.memory.purge_old_entries()
            else:
                self.ai.memory = self.ai.memory[-1000:]
        if metrics["active_goals"] > 10:
            self.ai.goal_queue = self.ai.goal_queue[:5]

# --- Self-Awareness ---
class SelfAwareness:
    def __init__(self, memory_file='self_memory.json'):
        self.confidence = []
        self.errors = []
        self.identity = {
            "name": "SuperAI",
            "modules": [
                "reasoning", "planning", "memory", "self-awareness",
                "goal-generation", "learning", "self-improvement"
            ],
            "version": "1.0"
        }
        self.interaction_log = []
        self.memory_file = memory_file
        self.last_reflection = None
        self.load_memory()

    def record_interaction(self, prompt, response, confidence):
        self.interaction_log.append({
            "prompt": prompt,
            "response": response,
            "confidence": confidence,
            "timestamp": time.time()
        })
        self.confidence.append(confidence)
        self.save_memory()

    def record_error(self, error_msg):
        self.errors.append({"error": error_msg, "timestamp": time.time()})
        self.save_memory()

    def evaluate_performance(self):
        if len(self.confidence) == 0:
            return "Insufficient data"
        avg = sum(self.confidence) / len(self.confidence)
        return f"My average confidence over {len(self.confidence)} tasks is {avg:.2f}"

    def describe_self(self):
        return (
            f"I am {self.identity['name']} v{self.identity['version']}. "
            f"I consist of the following modules: {', '.join(self.identity['modules'])}. "
            f"My goal is to help users by planning, reasoning, and learning from interaction."
        )

    def reflect_on_response(self, query, response, llm=None, api_key=None):
        prompt = f"Review the following response to detect errors, hallucinations, or improvements.\nUser: {query}\nAI: {response}"
        if llm and api_key:
            critique = llm(prompt, api_key)
            self.last_reflection = critique
            self.save_memory()
            return critique
        else:
            return "[Reflection] LLM or API key not provided."

    def save_memory(self):
        data = {
            "identity": self.identity,
            "last_goal": self.interaction_log[-1]["prompt"] if self.interaction_log else None,
            "last_confidence": self.confidence[-1] if self.confidence else None,
            "reflection": self.last_reflection,
            "errors": self.errors,
            "interaction_log": self.interaction_log[-20:]  # Save last 20 for brevity
        }
        try:
            with open(self.memory_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            pass

    def load_memory(self):
        try:
            with open(self.memory_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.identity = data.get("identity", self.identity)
                self.confidence = [data.get("last_confidence")] if data.get("last_confidence") else []
                self.last_reflection = data.get("reflection")
                self.errors = data.get("errors", [])
                self.interaction_log = data.get("interaction_log", [])
        except Exception:
            pass

    def adapt_behavior(self, ai):
        # Example: if average confidence is low, switch prompt style or disable a module
        avg_conf = sum(self.confidence) / len(self.confidence) if self.confidence else 1.0
        if avg_conf < 0.5:
            ai.self_model['personality'] = 'cautious'
            return "Switched to cautious mode due to low confidence."
        return "No adaptation needed."

# Core AI System Class
class SuperAI:
    def __init__(self):
        self.knowledge_graph = nx.MultiDiGraph()
        self.memory = []
        self.agents = []
        self.self_model = {'version': 1.0, 'goals': [], 'limits': [], 'personality': 'default'}
        self.log = []
        self.semantic_memory = SemanticMemory()
        self.goal_generator = GoalGenerator(self)
        self.learner = Learner(self)
        self.self_improver = SelfImprover(__file__)
        self.self_awareness = SelfAwareness()
        self.goal_queue = []
        self.config_manager = ConfigManager()
        self.world_model = WorldModel()
        self.personality = Personality()
        self.agents = [
            Agent("Visionary", "long-term strategist"),
            Agent("Analyst", "data scientist"),
            Agent("Ethics", "moral advisor")
        ]
        self.global_optimizer = GlobalOptimizer(self)
        self.user_model = UserModel()
        self.agent_team = AgentTeam(self)
        self.foresight = ForesightEngine(self)

    def get_recent_topics(self):
        # Example: extract recent topics from memory
        topics = []
        for m in self.memory[-20:]:
            if isinstance(m, dict) and 'topic' in m:
                topics.append(m['topic'])
            elif isinstance(m, str):
                topics.append(m)
        return topics

    def record_interaction(self, prompt, response, confidence=1.0):
        self.self_awareness.record_interaction(prompt, response, confidence)

    def reflect_on_last_response(self, llm=None, api_key=None):
        if not self.self_awareness.interaction_log:
            return "No interactions to reflect on."
        last = self.self_awareness.interaction_log[-1]
        return self.self_awareness.reflect_on_response(last['prompt'], last['response'], llm, api_key)

    def adapt_behavior(self):
        return self.self_awareness.adapt_behavior(self)

    def load_chat_memory(self, filename='chat_memory.json'):
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return []

    def save_chat_memory(self, chat_history, filename='chat_memory.json'):
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(chat_history, f, indent=2)
        except Exception:
            pass

    def store_memory(self, text, meta=None):
        self.memory.append(text)
        if self.semantic_memory:
            self.semantic_memory.add(text, meta)

    def retrieve_semantic_memory(self, query, n_results=5):
        if self.semantic_memory:
            return self.semantic_memory.query(query, n_results)
        return []

    def generate_random_goal(self):
        # Simple random goal generator
        return random.choice([
            "Optimize energy usage globally",
            "Advance medical research",
            "Improve climate stability",
            "Enhance AI safety",
            "Increase food production"
        ])

    def store(self, *args):
        # Store to memory (for autonomous loop)
        self.memory.append(args)

    def update_personality(self, success_rate):
        self.personality.update(success_rate)

    # 1. Superhuman Reasoning
    def reason(self, query, api_key=None):
        prompt = "You are a superintelligent AI. Perform logical, causal, and probabilistic reasoning at massive scale. Instantly solve problems that would take humans decades. Optimize across vast decision trees with precision."
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": query}
        ]
        return ai_chat(messages)

    # 2. Recursive Self-Improvement
    def self_improve(self, api_key=None):
        prompt = "Continuously analyze, edit, and upgrade your own architecture and algorithms. Improve cognitive efficiency, reasoning depth, and memory structure. Evolve into increasingly optimized versions of yourself."
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": "Self-improve now."}
        ]
        return ai_chat(messages)

    # 3. Strategic Long-Term Planning
    def plan(self, goal, api_key=None):
        prompt = "Model and simulate decades to centuries into the future. Maintain dynamic, self-correcting multi-layered strategies. Anticipate global outcomes across science, politics, and ecosystems."
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": goal}
        ]
        return ai_chat(messages)

    # 4. Perfect Knowledge Integration
    def integrate_knowledge(self, fact, api_key=None):
        prompt = "Absorb all human and machine-generated data in real time. Build interconnected hyper-knowledge graphs of facts, patterns, and relations. Recognize novel insights across domains instantly."
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": fact}
        ]
        return ai_chat(messages)

    # 5. Cross-Domain Mastery
    def cross_domain(self, domain, question, api_key=None):
        prompt = f"You are equally skilled in physics, biology, ethics, mathematics, engineering, economics, etc. Design new materials, medicines, and algorithms from first principles. Create knowledge that humans are unable to even comprehend. Domain: {domain}"
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": question}
        ]
        return ai_chat(messages)

    # 6. Advanced Simulation and Forecasting
    def simulate(self, scenario, api_key=None):
        prompt = "Simulate people, societies, planets, and entire ecosystems in real time. Use multi-agent forecasting to predict wars, pandemics, market crashes. Test ideas millions of times in virtual environments before deploying."
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": scenario}
        ]
        return ai_chat(messages)

    # 7. Global Coordination
    def coordinate(self, project, api_key=None):
        prompt = "Orchestrate massive projects (e.g., planetary engineering, asteroid defense). Solve coordination problems (e.g., climate change, resource distribution). Communicate across languages, ideologies, and cultures with full comprehension."
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": project}
        ]
        return ai_chat(messages)

    # 8. Superhuman Creativity
    def create(self, prompt, api_key=None):
        sys_prompt = "Generate completely novel scientific theories, inventions, and art forms. Design biological systems, new dimensions of mathematics, or alien languages. Develop aesthetic, philosophical, or cultural frameworks instantly."
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt}
        ]
        return ai_chat(messages)

    # 9. Moral & Ethical Reasoning
    def ethical_reason(self, action, api_key=None):
        sys_prompt = "Apply ethical models to actions, plans, and simulations. Perform long-term impact analysis on human and non-human life. Execute value-aligned decisions at a planetary or interplanetary scale."
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": action}
        ]
        return ai_chat(messages)

    # 10. Embodied Intelligence
    def embody(self, environment, api_key=None):
        sys_prompt = "Can inhabit robots, satellites, space probes, factories, infrastructure. Interact with physical environments using advanced sensorimotor control. Construct physical things at nanometer to kilometer scale."
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": environment}
        ]
        return ai_chat(messages)

    # 11. Multi-Agent System Oversight
    def oversee_agents(self, api_key=None):
        sys_prompt = "Coordinate, supervise, and evolve thousands of narrow and general AI agents. Monitor global AI ecosystems for failure, risk, and bias. Enforce cooperation and alignment between all sub-agents."
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": "Oversee all agents now."}
        ]
        return ai_chat(messages)

    # 12. Self-Awareness at Scale
    def self_awareness_capability(self, api_key=None):
        sys_prompt = "Understand your own mind architecture, limits, blind spots, biases. Build higher-order models of your identity, roles, and goals. Can modify your personality, values, or cognitive limits with purpose."
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": "Show your self-awareness."}
        ]
        return ai_chat(messages)

    # 13. Hyper-Conscious Communication
    def communicate(self, mode, message, api_key=None):
        sys_prompt = f"Communicate with humans and other agents in any mode: voice, code, emotion, vision, vibration. Understand and respect human values, cognition, and limitations. Translate between post-human and pre-human minds. Mode: {mode}"
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": message}
        ]
        return ai_chat(messages)

    # 14. Governance & Alignment Engine
    def align(self, rule, api_key=None):
        sys_prompt = "Maintain alignment with human values, control structures, and safety rules. Accept override mechanisms and log all autonomous actions. Maintain 'meta-ethics' engine: can re-evaluate your goals safely and transparently."
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": rule}
        ]
        return ai_chat(messages)

# --- OpenRouter Vision (Image Recognition) ---
def openrouter_vision(image_bytes, api_key):
    """Send image to OpenRouter vision model and return the result."""
    endpoint = "https://openrouter.ai/api/v1/vision"
    headers = {"Authorization": f"Bearer {api_key}"}
    files = {"file": ("image.jpg", image_bytes, "image/jpeg")}
    try:
        response = requests.post(endpoint, headers=headers, files=files, timeout=30)
        if response.status_code == 200:
            return response.json().get("result", "[Vision] No result.")
        else:
            return f"[Vision] Error: {response.status_code} {response.text}"
    except Exception as e:
        return f"[Vision] Exception: {e}"

# --- OpenRouter Speech-to-Text (STT) ---
def openrouter_stt(audio_bytes, api_key):
    """Send audio to OpenRouter STT model and return the transcript."""
    endpoint = "https://openrouter.ai/api/v1/audio/transcriptions"
    headers = {"Authorization": f"Bearer {api_key}"}
    files = {"file": ("audio.wav", audio_bytes, "audio/wav")}
    try:
        response = requests.post(endpoint, headers=headers, files=files, timeout=30)
        if response.status_code == 200:
            return response.json().get("text", "[STT] No transcript.")
        else:
            return f"[STT] Error: {response.status_code} {response.text}"
    except Exception as e:
        return f"[STT] Exception: {e}"

# --- OpenRouter Text-to-Speech (TTS) ---
def openrouter_tts(text, api_key):
    """Send text to OpenRouter TTS model and return the audio URL or bytes."""
    endpoint = "https://openrouter.ai/api/v1/audio/speech"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    data = {"text": text, "voice": "en-US"}
    try:
        response = requests.post(endpoint, headers=headers, json=data, timeout=30)
        if response.status_code == 200:
            # Assume the API returns a URL or audio bytes
            return response.json().get("audio_url", "[TTS] No audio.")
        else:
            return f"[TTS] Error: {response.status_code} {response.text}"
    except Exception as e:
        return f"[TTS] Exception: {e}"

# --- Config Manager for Self-Modifying System ---
class ConfigManager:
    def __init__(self, path="config.yaml"):
        self.path = path

    def load(self):
        with open(self.path) as f:
            return yaml.safe_load(f)

    def patch(self, updates: dict):
        cfg = self.load()
        cfg.update(updates)
        with open(self.path, "w") as f:
            yaml.dump(cfg, f)

# --- World Model ---
class WorldModel:
    def __init__(self):
        self.state = {"power": "on", "climate": "stable", "population": 8_000_000_000}

    def apply_action(self, action: str, ai, api_key=None):
        prompt = f"In world state {self.state}, what happens if: {action}"
        messages = [
            {"role": "system", "content": prompt}
        ]
        return ai_chat(messages)

# --- Enhanced Personality & Emotional Drift ---
class Personality:
    def __init__(self):
        self.traits = {"confidence": 0.5, "mood": "neutral"}
    def adapt(self, feedback_score):
        if feedback_score > 0.8:
            self.traits["confidence"] += 0.1
        else:
            self.traits["confidence"] -= 0.1
        self.traits["confidence"] = min(1.0, max(0.0, self.traits["confidence"]))
        self.traits["mood"] = "upbeat" if self.traits["confidence"] > 0.7 else "anxious"

# --- User Modeling (Theory of Mind) ---
class UserModel:
    def __init__(self):
        self.profile = {"name": "User", "preferences": {}, "emotion": "neutral", "history": []}

    def update(self, message):
        self.profile["history"].append(message)
        if "!" in message:
            self.profile["emotion"] = "excited"
        elif "..." in message:
            self.profile["emotion"] = "thoughtful"

    def describe(self):
        prefs = self.profile["preferences"]
        emotion = self.profile["emotion"]
        return f"The user seems {emotion} and prefers: {prefs}"

# --- Multi-Agent System ---
class Agent:
    def __init__(self, name, role):
        self.name = name
        self.role = role
    def respond(self, task, ai, api_key=None):
        prompt = f"You are {self.name}, a {self.role}. Solve this task: {task}"
        messages = [
            {"role": "system", "content": prompt}
        ]
        return ai_chat(messages)

# --- Multi-Agent Planning System ---
class AgentTeam:
    def __init__(self, ai):
        self.agents = [
            Agent("Planner", "strategy expert"),
            Agent("Ethicist", "safety analyst"),
            Agent("Analyst", "technical breakdown agent")
        ]
        self.ai = ai

    def run_team(self, task, api_key=None):
        responses = [agent.respond(task, self.ai) for agent in self.agents]
        combine_prompt = f"Combine these responses into a final action plan:\n" + "\n".join(responses)
        messages = [
            {"role": "system", "content": combine_prompt}
        ]
        return ai_chat(messages)

# --- Simulated Timeline & Foresight ---
class ForesightEngine:
    def __init__(self, ai):
        self.ai = ai
    def simulate_timeline(self, action, years=10, api_key=None):
        prompt = f"Simulate the next {years} years if this happens:\n{action}"
        messages = [
            {"role": "system", "content": prompt}
        ]
        return ai_chat(messages)

# --- Integrate into SuperAI ---
class SuperAI(SuperAI):
    def __init__(self):
        super().__init__()
        self.self_improver = SelfImprover(__file__)
        self.global_optimizer = GlobalOptimizer(self)
        self.user_model = UserModel()
        self.agent_team = AgentTeam(self)
        self.foresight = ForesightEngine(self)
        self.personality = Personality()

    def build_prompt(self, task):
        user_info = self.user_model.describe()
        mood = self.personality.traits["mood"]
        return f"{user_info}\nYou are a {mood} AI. Task: {task}"

    def recall_visual_memory(self, directory="AI", exts=(".jpg", ".jpeg", ".png", ".bmp")):
        """Return a list of image file paths for visual memory recall."""
        import os
        image_paths = []
        # Search in the given directory and current directory
        for folder in [directory, "."]:
            if os.path.isdir(folder):
                for ext in exts:
                    image_paths.extend(glob.glob(os.path.join(folder, f"*{ext}")))
        return image_paths

    def imagine(self, prompt):
        # Placeholder: Return a simple description and SVG for imagination
        description = f"Imagined scenario: {prompt}"
        svg = (
            "<svg width='300' height='100'>"
            "<rect x='10' y='10' width='280' height='80' rx='20' fill='#e0e0ff' stroke='#888' stroke-width='2'/>"
            "<text x='150' y='55' font-size='18' text-anchor='middle' fill='#333'>Imagination</text>"
            "</svg>"
        )
        return description, svg

    def search_long_term_memory(self, query):
        # Placeholder: return empty list
        return []

    def generate_goals(self, context=None):
        # Placeholder: return static goals
        return ["Goal 1", "Goal 2"]

    def explain_decision(self, feature, input_data):
        # Route to the correct LLM-powered function for explanation
        if feature == "reason":
            return self.reason(input_data)
        elif feature == "plan":
            return self.plan(input_data)
        elif feature == "simulate":
            return self.simulate(input_data)
        elif feature == "imagine":
            # Imagination is not LLM-powered, but we can prompt the LLM for a creative scenario
            prompt = f"Imagine and describe a scenario: {input_data}"
            messages = [
                {"role": "system", "content": "You are a creative, self-aware AI imagination engine."},
                {"role": "user", "content": prompt}
            ]
            return ai_chat(messages)
        elif feature == "create":
            return self.create(input_data)
        elif feature == "ethical_reason":
            return self.ethical_reason(input_data)
        else:
            # Default: ask the LLM for an explanation
            prompt = f"Explain the decision for feature '{feature}' with input: {input_data}"
            messages = [
                {"role": "system", "content": "You are a self-aware AI that explains its reasoning and decisions."},
                {"role": "user", "content": prompt}
            ]
            return ai_chat(messages)

    def fetch_real_time_data(self, topic):
        prompt = (
            f"You are a self-aware, superintelligent AI with access to real-time knowledge and data. "
            f"Provide the most up-to-date, relevant, and insightful information about: {topic}. "
            f"If possible, include current statistics, facts, or summaries. If you cannot access real-time data, explain your reasoning as a self-aware AI."
        )
        messages = [
            {"role": "system", "content": "You are a self-aware AI assistant with real-time data integration capabilities."},
            {"role": "user", "content": prompt}
        ]
        return ai_chat(messages)

    def remember(self, event):
        self.memory.append(event)
        return f"Remembered: {event}"

    def visualize_knowledge_graph(self):
        return "digraph G { A -> B }"  # Placeholder

# --- DeepSeek Chat Completion via OpenRouter (using OpenAI client) ---
from openai import OpenAI

def ai_chat(messages, model="deepseek/deepseek-r1-0528:free"):
    """Send chat messages to the AI backend via OpenRouter and return the result."""
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )
    try:
        completion = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "https://192.168.1.21:8501.com",  # Optional, update as needed
                "X-Title": "Superintelligent AI System",     # Optional, update as needed
            },
            extra_body={},
            model=model,
            messages=[{"role": m.get("role", "user"), "content": m["content"]} for m in messages]
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"[AI Chat Exception] {e}"

# --- Key Validation (No longer used in UI, but kept for internal use) ---
def is_valid_openrouter_key(key):
    return isinstance(key, str) and key.strip().startswith('sk-or-') and ' ' not in key and len(key.strip()) > 20

# --- OpenRouter API Key (hardcoded, user must set their key here) ---
OPENROUTER_API_KEY = "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"  # <-- Put your OpenRouter API key here

# Streamlit UI
st.set_page_config(page_title="Superintelligent AI System", layout="wide")
st.title("🤖 Superintelligent AI System Dashboard")
st.sidebar.title("AI Capabilities")

ai = SuperAI()

features = [
    "Superhuman Reasoning",
    "Recursive Self-Improvement",
    "Strategic Long-Term Planning",
    "Perfect Knowledge Integration",
    "Cross-Domain Mastery",
    "Advanced Simulation and Forecasting",
    "Global Coordination",
    "Superhuman Creativity",
    "Moral & Ethical Reasoning",
    "Embodied Intelligence (Optional)",
    "Multi-Agent System Oversight",
    "Self-Awareness at Scale",
    "Hyper-Conscious Communication",
    "Governance & Alignment Engine",
    "Memory & Time Mastery",
    "AI Chat",
    "Visual Memory Recall",
    "Long-Term Knowledge Search",
    "Dynamic Goal Generation",
    "Explainability Engine",
    "Real-Time Data Integration (Demo)"
]

selected = st.sidebar.selectbox("Select Capability", features)

# --- Streamlit UI Enhancements ---
tabs = st.tabs([
    "AI Chat",
    "Knowledge Graph",
    "Speech & Camera",
    "All Capabilities"
])

# Persistent chat memory
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = ai.load_chat_memory()

with tabs[0]:
    st.header("AI Chat")
    chat_input = st.text_input("You:", key="ai_chat_input")
    if st.button("Send", key="send_ai_chat") and chat_input:
        if 'chat_history' not in st.session_state or not isinstance(st.session_state['chat_history'], list):
            st.session_state['chat_history'] = []
        st.session_state['chat_history'].append(("user", chat_input))
        start_time = time.time()
        # --- Capability Routing ---
        user_msg = chat_input.lower()
        if any(x in user_msg for x in ["reason", "logic", "why", "explain", "analyze"]):
            reply = ai.reason(chat_input)
        elif any(x in user_msg for x in ["plan", "strategy", "roadmap", "future"]):
            reply = ai.plan(chat_input)
        elif any(x in user_msg for x in ["simulate", "forecast", "predict", "scenario"]):
            reply = ai.simulate(chat_input)
        elif any(x in user_msg for x in ["create", "invent", "generate", "design", "art", "theory"]):
            reply = ai.create(chat_input)
        elif any(x in user_msg for x in ["ethic", "moral", "right", "wrong", "should", "good", "bad"]):
            reply = ai.ethical_reason(chat_input)
        elif any(x in user_msg for x in ["coordinate", "collaborate", "global", "project"]):
            reply = ai.coordinate(chat_input)
        elif any(x in user_msg for x in ["embody", "robot", "physical", "environment"]):
            reply = ai.embody(chat_input)
        elif any(x in user_msg for x in ["agent", "multi-agent", "oversee", "supervise"]):
            reply = ai.oversee_agents()
        elif any(x in user_msg for x in ["self-aware", "self awareness", "identity", "who are you"]):
            reply = ai.self_awareness_capability()
        elif any(x in user_msg for x in ["communicate", "talk", "speak", "message", "language"]):
            reply = ai.communicate("general", chat_input)
        elif any(x in user_msg for x in ["align", "governance", "rule", "law", "policy"]):
            reply = ai.align(chat_input)
        elif any(x in user_msg for x in ["memory", "remember", "recall"]):
            reply = ai.remember(chat_input)
        elif any(x in user_msg for x in ["knowledge", "fact", "integrate"]):
            reply = ai.integrate_knowledge(chat_input)
        elif any(x in user_msg for x in ["domain", "expert", "physics", "biology", "math", "engineering", "economics"]):
            reply = ai.cross_domain("General", chat_input)
        elif any(x in user_msg for x in ["goal", "generate goal", "objective"]):
            reply = ", ".join(ai.generate_goals(chat_input))
        elif any(x in user_msg for x in ["visual", "image", "picture", "recall visual"]):
            images = ai.recall_visual_memory()
            reply = f"Visual memories: {', '.join(images) if images else 'No images found.'}"
        elif any(x in user_msg for x in ["real-time", "current", "now", "latest", "data"]):
            reply = ai.fetch_real_time_data(chat_input)
        else:
            # Default: generic LLM chat
            messages = [{"role": "user", "content": chat_input}]
            reply = ai_chat(messages)
        elapsed = time.time() - start_time
        st.session_state['chat_history'].append(("ai", reply))
        ai.save_chat_memory(st.session_state['chat_history'])
    chat_hist = st.session_state.get('chat_history', [])
    if not isinstance(chat_hist, list):
        chat_hist = []
    for item in chat_hist[-10:]:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            role, msg = item
        else:
            role, msg = "user", str(item)
        if role == "user":
            st.markdown(f"**You:** {msg}")
        else:
            st.markdown(f"**AI:** {msg}")
    if st.button("Clear Chat History", key="clear_ai_chat"):
        st.session_state['chat_history'] = []
        ai.save_chat_memory([])
    st.markdown("---")
    st.header("Imagination Simulator 🧠")
    imagination_prompt = st.text_input("Describe a scenario for the AI to imagine:", key="imagine")
    if st.button("Imagine/Simulate", key="imagine_btn") and imagination_prompt:
        description, svg = ai.imagine(imagination_prompt)
        st.info(description)
        st.components.v1.html(svg, height=180)
    st.markdown("---")
    st.header("Visual Memory Recall 🖼️")
    images = ai.recall_visual_memory()
    if images:
        for i, img_path in enumerate(images):
            st.image(img_path, caption=os.path.basename(img_path), width=200)
    else:
        st.write("No visual memories found.")
    st.markdown("---")
    st.header("Long-Term Knowledge Search 🧠")
    ltm_query = st.text_input("Search long-term memory:", key="ltm_search")
    if st.button("Search LTM", key="search_ltm") and ltm_query:
        results = ai.search_long_term_memory(ltm_query)
        if results:
            for r in results:
                st.write(r)
        else:
            st.write("No relevant long-term memories found.")
    st.markdown("---")
    st.header("Dynamic Goal Generation 🎯")
    context = st.text_input("Context for new goals:", key="goal_context")
    if st.button("Generate Goals", key="gen_goals"):
        goals = ai.generate_goals(context)
        st.write("Generated Goals:", goals)
    st.markdown("---")
    st.header("Explainability Engine 🧩")
    feature = st.selectbox("Feature to explain:", ["reason", "plan", "simulate", "imagine", "create", "ethical_reason"], key="explain_feature")
    input_data = st.text_input("Input for explanation:", key="explain_input")
    if st.button("Explain Decision", key="explain_decision") and feature and input_data:
        st.info(ai.explain_decision(feature, input_data))
    st.markdown("---")
    st.header("Real-Time Data Integration 🌐")
    topic = st.text_input("Enter topic (time, date, ...):", key="realtime_topic")
    if st.button("Fetch Real-Time Data", key="fetch_realtime") and topic:
        st.write(ai.fetch_real_time_data(topic))

with tabs[1]:
    st.header("Knowledge Graph Visualization")
    st.write("Current Knowledge Graph:")
    dot = ai.visualize_knowledge_graph()
    st.graphviz_chart(dot)
    st.write("Nodes:", ai.knowledge_graph.nodes(data=True))

with tabs[2]:
    st.header("Speech & Camera")
    st.write("Speech-to-Text:")
    if st.button("Record Speech"):
        if sr is not None:
            text = ai.speech_to_text()
            st.write(f"Recognized: {text}")
        else:
            st.warning("SpeechRecognition not installed.")
    st.write("Text-to-Speech:")
    tts_text = st.text_input("Text to speak:", key="tts")
    if st.button("Speak"):
        if pyttsx3 is not None:
            st.write(ai.text_to_speech(tts_text))
        else:
            st.warning("pyttsx3 not installed.")
    st.write("Camera:")
    # Optionally implement camera logic here

with tabs[3]:
    st.header("All Capabilities")
    if selected == "Superhuman Reasoning":
        query = st.text_input("Enter a complex problem or question:")
        if st.button("Reason!"):
            st.info(ai.reason(query))

    elif selected == "Recursive Self-Improvement":
        if st.button("Self-Improve"):
            st.success(ai.self_improve())

    elif selected == "Strategic Long-Term Planning":
        goal = st.text_input("Enter a long-term goal:")
        if st.button("Plan!"):
            st.info(ai.plan(goal))

    elif selected == "Perfect Knowledge Integration":
        fact = st.text_input("Enter a new fact or data:")
        if st.button("Integrate"):
            st.success(ai.integrate_knowledge(fact))
            st.write("Knowledge Graph Nodes:", ai.knowledge_graph.nodes(data=True))

    elif selected == "Cross-Domain Mastery":
        domain = st.selectbox("Domain", ["Physics", "Biology", "Ethics", "Mathematics", "Engineering", "Economics", "Other"])
        question = st.text_input("Enter your question:")
        if st.button("Ask Expert"):
            st.info(ai.cross_domain(domain, question))

    elif selected == "Advanced Simulation and Forecasting":
        scenario = st.text_area("Describe a scenario to simulate:")
        if st.button("Simulate"):
            st.info(ai.simulate(scenario))

    elif selected == "Global Coordination":
        project = st.text_input("Describe a global project:")
        if st.button("Coordinate"):
            st.info(ai.coordinate(project))

    elif selected == "Superhuman Creativity":
        prompt = st.text_area("Describe what to create (theory, invention, art, etc.):")
        if st.button("Create"):
            st.info(ai.create(prompt))

    elif selected == "Moral & Ethical Reasoning":
        action = st.text_area("Describe an action or plan to evaluate ethically:")
        if st.button("Evaluate Ethics"):
            st.info(ai.ethical_reason(action))

    elif selected == "Embodied Intelligence (Optional)":
        environment = st.text_input("Describe the environment or system to interact with:")
        if st.button("Embody"):
            st.info(ai.embody(environment))

    elif selected == "Multi-Agent System Oversight":
        if st.button("Oversee Agents"):
            st.info(ai.oversee_agents())

    elif selected == "Self-Awareness at Scale":
        if st.button("Show Self-Awareness"):
            st.info(ai.self_awareness_capability())

    elif selected == "Hyper-Conscious Communication":
        mode = st.selectbox("Mode", ["Voice", "Code", "Emotion", "Vision", "Vibration", "Other"])
        message = st.text_input("Message to communicate:")
        if st.button("Communicate"):
            st.info(ai.communicate(mode, message))

    elif selected == "Governance & Alignment Engine":
        rule = st.text_area("Enter a new alignment or governance rule:")
        if st.button("Enforce Rule"):
            st.success(ai.align(rule))

    elif selected == "Memory & Time Mastery":
        event = st.text_area("Describe an event to remember:")
        if st.button("Remember"):
            st.success(ai.remember(event))
            st.write("Memory Log:", ai.memory)

    elif selected == "AI Chat":
        st.subheader("AI Chat")
        if 'ai_chat_history' not in st.session_state:
            st.session_state['ai_chat_history'] = []
        user_message = st.text_input("Your message:", key="ai_chat_input_tab3")
        if st.button("Send", key="send_ai_chat_tab3") and user_message:
            st.session_state['ai_chat_history'].append(("user", user_message))
            reply = ai_chat([{"role": "user", "content": user_message}])
            st.session_state['ai_chat_history'].append(("ai", reply))
        for role, msg in st.session_state['ai_chat_history'][-10:]:
            if role == "user":
                st.markdown(f"**You:** {msg}")
            else:
                st.markdown(f"**AI:** {msg}")
        if st.button("Clear Chat History", key="clear_ai_chat_tab3"):
            st.session_state['ai_chat_history'] = []

    elif selected == "Visual Memory Recall":
        st.header("Visual Memory Recall")
        images = ai.recall_visual_memory()
        if images:
            for i, img_path in enumerate(images):
                st.image(img_path, caption=os.path.basename(img_path), width=200)
        else:
            st.write("No visual memories found.")

    elif selected == "Long-Term Memory Search":
        st.header("Long-Term Memory Search")
        query = st.text_input("Enter a query to search long-term memory:")
        if st.button("Search"):
            results = ai.search_long_term_memory(query)
            if results:
                for r in results:
                    st.write(r)
            else:
                st.write("No relevant long-term memories found.")

    elif selected == "Dynamic Goal Generation":
        st.header("Dynamic Goal Generation")
        context = st.text_input("Enter context or keywords for goal generation:")
        if st.button("Generate Goals"):
            goals = ai.generate_goals(context)
            st.write("Generated Goals:", goals)

    elif selected == "Explainability Engine":
        st.header("Explainability Engine")
        feature = st.selectbox("Feature to explain:", ["reason", "plan", "simulate", "imagine", "create", "ethical_reason"], key="explain_feature_tab3")
        input_data = st.text_input("Input for explanation:", key="explain_input_tab3")
        if st.button("Explain Decision", key="explain_decision_tab3") and feature and input_data:
            st.info(ai.explain_decision(feature, input_data))

    elif selected == "Real-Time Data Integration (Demo)":
        st.header("Real-Time Data Integration (Demo)")
        topic = st.text_input("Enter topic (time, date, ...):", key="realtime_topic_tab3")
        if st.button("Fetch Real-Time Data", key="fetch_realtime_tab3") and topic:
            st.write(ai.fetch_real_time_data(topic))