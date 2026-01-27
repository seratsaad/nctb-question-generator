"""
NCTB Question Generator - Streamlit Web App
With AI-generated figures (Matplotlib + SVG) and tables
"""

import streamlit as st
import os
from typing import List, Dict, Optional, Tuple
import re
from io import BytesIO
import base64

# Figure Generation
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle, Rectangle, FancyArrowPatch, Arc
import numpy as np

# Vector Store
import chromadb
from chromadb.utils import embedding_functions

# PDF Generation
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors

# LLM APIs
from openai import OpenAI

# Page config
st.set_page_config(page_title="NCTB Question Generator", page_icon="📚", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: bold; color: #1E88E5; text-align: center; margin-bottom: 2rem; }
    .sub-header { font-size: 1.2rem; color: #666; text-align: center; margin-bottom: 2rem; }
    .stButton > button { width: 100%; background-color: #1E88E5; color: white; }
    .question-box { padding: 1.5rem; background-color: #F5F5F5; border-radius: 0.5rem; margin: 1rem 0; }
    .figure-box { background-color: white; padding: 1rem; border-radius: 0.5rem; border: 1px solid #ddd; margin: 1rem 0; text-align: center; }
</style>
""", unsafe_allow_html=True)

# Curriculum data
CURRICULUM_DATA = {
    "Class 9-10": {
        "Physics": {
            "chapters": [
                "Chapter 1: Physical Quantities and Measurement",
                "Chapter 2: Motion",
                "Chapter 3: Force",
                "Chapter 4: Work, Power and Energy",
                "Chapter 5: Matter: Structure and Properties",
                "Chapter 6: Effects of Heat on Matter",
                "Chapter 7: Waves and Sound",
                "Chapter 8: Light",
                "Chapter 9: Electricity",
                "Chapter 10: Magnetic Effects of Electric Current",
                "Chapter 11: Electronics",
                "Chapter 12: Modern Physics and Radioactivity",
            ]
        }
    }
}


# ============== FIGURE GENERATORS ==============

class MatplotlibFigureGenerator:
    """Generates technical physics diagrams using matplotlib"""
    
    def parse_spec(self, text: str) -> Tuple[Optional[str], Dict]:
        """Parse [FIGURE: type | params] format"""
        match = re.search(r'\[FIGURE:\s*(\w+)(?:\s*\|\s*([^\]]+))?\]', text, re.IGNORECASE)
        if not match:
            return None, {}
        
        fig_type = match.group(1).lower()
        params = {}
        
        if match.group(2):
            for p in match.group(2).split(','):
                if '=' in p:
                    k, v = p.split('=', 1)
                    k, v = k.strip().lower(), v.strip()
                    # Convert types
                    if v.lower() == 'true': v = True
                    elif v.lower() == 'false': v = False
                    elif re.match(r'^-?\d+\.?\d*$', v): v = float(v) if '.' in v else int(v)
                    params[k] = v
        
        return fig_type, params
    
    def generate(self, spec: str) -> Optional[BytesIO]:
        """Generate figure from spec"""
        fig_type, params = self.parse_spec(spec)
        if not fig_type:
            return None
        
        generators = {
            'circuit': self._circuit,
            'circuit_series': lambda p: self._circuit({**p, 'series': True}),
            'circuit_parallel': lambda p: self._circuit({**p, 'series': False}),
            'wave': self._wave,
            'motion_graph': self._motion_graph,
            'vt_graph': lambda p: self._motion_graph({**p, 'graph_type': 'v-t'}),
            'st_graph': lambda p: self._motion_graph({**p, 'graph_type': 's-t'}),
            'force_diagram': self._force_diagram,
            'fbd': self._force_diagram,
            'pendulum': self._pendulum,
            'projectile': self._projectile,
            'ray_diagram': self._ray_diagram,
            'lens': self._ray_diagram,
            'pulley': self._pulley,
            'spring': self._spring,
        }
        
        try:
            if fig_type in generators:
                return generators[fig_type](params)
            return self._placeholder(fig_type, params)
        except Exception as e:
            return self._placeholder(fig_type, params, str(e))
    
    def _save_fig(self, fig) -> BytesIO:
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        buf.seek(0)
        return buf
    
    def _placeholder(self, fig_type: str, params: Dict, error: str = None) -> BytesIO:
        fig, ax = plt.subplots(figsize=(6, 3))
        text = f"[Figure: {fig_type}]"
        if error: text += f"\nError: {error}"
        ax.text(0.5, 0.5, text, ha='center', va='center', fontsize=11,
                bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='orange'))
        ax.axis('off')
        return self._save_fig(fig)
    
    def _circuit(self, p: Dict) -> BytesIO:
        fig, ax = plt.subplots(figsize=(8, 4))
        n = p.get('resistors', 2)
        v = p.get('voltage', 12)
        series = p.get('series', True)
        
        if series:
            # Battery
            ax.plot([0.5, 0.5], [0.6, 0.9], 'k-', lw=2)
            ax.plot([0.3, 0.7], [0.9, 0.9], 'k-', lw=4)
            ax.plot([0.38, 0.62], [1.0, 1.0], 'k-', lw=2)
            ax.text(0.85, 0.95, f'{v}V', fontsize=10, fontweight='bold')
            
            # Top wire
            ax.plot([0.5, 3.5], [1.0, 1.0], 'k-', lw=2)
            ax.plot([3.5, 3.5], [1.0, 0.3], 'k-', lw=2)
            
            # Resistors
            spacing = 2.8 / n
            x = 0.5
            for i in range(n):
                ax.plot([x, x+0.1], [0.3, 0.3], 'k-', lw=2)
                # Zigzag
                zx, zy = [x+0.1], [0.3]
                w = spacing - 0.2
                for j in range(6):
                    zx.append(x + 0.1 + (j+0.5)*w/6)
                    zy.append(0.42 if j%2==0 else 0.18)
                zx.append(x + 0.1 + w)
                zy.append(0.3)
                ax.plot(zx, zy, 'k-', lw=2)
                ax.text(x + 0.1 + w/2, 0.05, f'R{i+1}', fontsize=10, ha='center')
                x += spacing
            
            ax.plot([x-spacing+0.1+w, 3.5], [0.3, 0.3], 'k-', lw=2)
            ax.plot([0.5, 0.5], [0.3, 0.6], 'k-', lw=2)
            ax.set_title('Series Circuit', fontweight='bold')
        else:
            # Parallel circuit
            ax.plot([0.5, 0.5], [0.8, 1.1], 'k-', lw=2)
            ax.plot([0.3, 0.7], [1.1, 1.1], 'k-', lw=4)
            ax.plot([0.38, 0.62], [1.2, 1.2], 'k-', lw=2)
            ax.text(0.85, 1.15, f'{v}V', fontsize=10, fontweight='bold')
            
            ax.plot([0.5, 3.2], [1.2, 1.2], 'k-', lw=2)
            ax.plot([0.5, 3.2], [0.3, 0.3], 'k-', lw=2)
            ax.plot([0.5, 0.5], [0.3, 0.8], 'k-', lw=2)
            ax.plot([3.2, 3.2], [0.3, 1.2], 'k-', lw=2)
            
            for i in range(n):
                x = 1.0 + i * 2.0 / (n+1)
                ax.plot([x, x], [1.2, 0.95], 'k-', lw=2)
                ax.add_patch(Rectangle((x-0.1, 0.55), 0.2, 0.4, fill=False, lw=2))
                ax.text(x, 0.4, f'R{i+1}', fontsize=10, ha='center')
                ax.plot([x, x], [0.55, 0.3], 'k-', lw=2)
            ax.set_title('Parallel Circuit', fontweight='bold')
        
        ax.set_xlim(0, 4)
        ax.set_ylim(0, 1.5)
        ax.set_aspect('equal')
        ax.axis('off')
        return self._save_fig(fig)
    
    def _wave(self, p: Dict) -> BytesIO:
        fig, ax = plt.subplots(figsize=(9, 4))
        wl = p.get('wavelength', 2)
        amp = p.get('amplitude', 1)
        cycles = p.get('cycles', 3)
        
        x = np.linspace(0, cycles * wl, 500)
        y = amp * np.sin(2 * np.pi * x / wl)
        ax.plot(x, y, 'b-', lw=2.5)
        ax.axhline(y=0, color='black', lw=1)
        
        # Wavelength arrow
        ax.annotate('', xy=(wl, amp+0.3), xytext=(0, amp+0.3),
                   arrowprops=dict(arrowstyle='<->', color='red', lw=2))
        ax.text(wl/2, amp+0.5, f'λ = {wl} m', fontsize=11, ha='center', color='red')
        
        # Amplitude arrow
        ax.annotate('', xy=(-0.2, amp), xytext=(-0.2, 0),
                   arrowprops=dict(arrowstyle='<->', color='green', lw=2))
        ax.text(-0.5, amp/2, f'A={amp}m', fontsize=10, ha='right', color='green', rotation=90, va='center')
        
        ax.set_xlabel('Distance (m)')
        ax.set_ylabel('Displacement (m)')
        ax.set_title('Wave Diagram', fontweight='bold')
        ax.grid(True, alpha=0.3)
        return self._save_fig(fig)
    
    def _motion_graph(self, p: Dict) -> BytesIO:
        fig, ax = plt.subplots(figsize=(8, 5))
        gtype = p.get('graph_type', p.get('type', 'v-t'))
        u = p.get('initial_velocity', 0)
        a = p.get('acceleration', 2)
        t_max = p.get('time', 10)
        
        t = np.linspace(0, t_max, 100)
        
        if 'v' in gtype.lower():
            y = u + a * t
            ylabel = 'Velocity (m/s)'
            title = 'Velocity-Time Graph'
            color = 'blue'
        elif 's' in gtype.lower():
            y = u * t + 0.5 * a * t**2
            ylabel = 'Displacement (m)'
            title = 'Displacement-Time Graph'
            color = 'green'
        else:
            y = np.ones_like(t) * a
            ylabel = 'Acceleration (m/s²)'
            title = 'Acceleration-Time Graph'
            color = 'red'
        
        ax.plot(t, y, color=color, lw=2.5)
        ax.fill_between(t, 0, y, alpha=0.2, color=color)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', lw=0.5)
        return self._save_fig(fig)
    
    def _force_diagram(self, p: Dict) -> BytesIO:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.add_patch(Rectangle((-0.4, -0.4), 0.8, 0.8, fill=True, 
                               facecolor='lightblue', edgecolor='black', lw=2))
        
        forces = p.get('forces', 'weight,normal')
        if isinstance(forces, str):
            forces = [f.strip().lower() for f in forces.split(',')]
        
        arrows = {
            'weight': ((0, -0.4), (0, -1.4), 'W', 'red'),
            'mg': ((0, -0.4), (0, -1.4), 'mg', 'red'),
            'normal': ((0, 0.4), (0, 1.4), 'N', 'green'),
            'friction': ((-0.4, 0), (-1.4, 0), 'f', 'orange'),
            'applied': ((0.4, 0), (1.4, 0), 'F', 'blue'),
            'tension': ((0, 0.4), (0, 1.5), 'T', 'purple'),
        }
        
        for f in forces:
            if f in arrows:
                s, e, lbl, c = arrows[f]
                ax.annotate('', xy=e, xytext=s, arrowprops=dict(arrowstyle='->', color=c, lw=2.5))
                ox = 0.15 if e[0] >= s[0] else -0.25
                oy = 0.15 if e[1] >= s[1] else -0.25
                ax.text(e[0]+ox, e[1]+oy, lbl, fontsize=12, color=c, fontweight='bold')
        
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title('Free Body Diagram', fontweight='bold')
        return self._save_fig(fig)
    
    def _pendulum(self, p: Dict) -> BytesIO:
        fig, ax = plt.subplots(figsize=(5, 6))
        L = p.get('length', 2)
        angle = p.get('angle', 30)
        
        # Support
        ax.plot([-0.8, 0.8], [3, 3], 'k-', lw=4)
        ax.fill_between([-0.8, 0.8], [3, 3], [3.15, 3.15], color='gray')
        
        # String and bob
        rad = np.radians(angle)
        bx, by = L * np.sin(rad), 3 - L * np.cos(rad)
        ax.plot([0, bx], [3, by], 'k-', lw=2)
        ax.add_patch(Circle((bx, by), 0.15, fill=True, facecolor='red', edgecolor='black', lw=2))
        
        # Vertical dashed line
        ax.plot([0, 0], [3, 3-L-0.3], 'k--', lw=1, alpha=0.5)
        
        # Angle arc
        arc = Arc((0, 3), 0.6, 0.6, angle=0, theta1=270, theta2=270+angle, color='blue', lw=1.5)
        ax.add_patch(arc)
        ax.text(0.25, 2.6, f'θ={angle}°', fontsize=10, color='blue')
        
        # Length label
        ax.text(bx/2 + 0.15, (3+by)/2, f'L={L}m', fontsize=10)
        
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(0, 3.5)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title('Simple Pendulum', fontweight='bold')
        return self._save_fig(fig)
    
    def _projectile(self, p: Dict) -> BytesIO:
        fig, ax = plt.subplots(figsize=(9, 5))
        v0 = p.get('velocity', 20)
        angle = p.get('angle', 45)
        g = 9.8
        
        rad = np.radians(angle)
        vx, vy = v0 * np.cos(rad), v0 * np.sin(rad)
        t_max = 2 * vy / g
        t = np.linspace(0, t_max, 100)
        x = vx * t
        y = vy * t - 0.5 * g * t**2
        
        ax.plot(x, y, 'b-', lw=2.5)
        ax.fill_between(x, 0, y, alpha=0.15)
        ax.axhline(y=0, color='black', lw=1)
        
        # Initial velocity vector
        ax.annotate('', xy=(vx*0.4, vy*0.4), xytext=(0, 0),
                   arrowprops=dict(arrowstyle='->', color='red', lw=2))
        ax.text(vx*0.4+1, vy*0.4, f'v₀={v0} m/s', fontsize=10, color='red')
        
        # Angle
        ax.text(2.5, 0.8, f'θ={angle}°', fontsize=10, color='green')
        
        # Max height and range
        h_max = vy**2 / (2*g)
        R = max(x)
        ax.axhline(y=h_max, color='orange', ls='--', alpha=0.7)
        ax.text(R*0.8, h_max+0.5, f'H={h_max:.1f}m', fontsize=9, color='orange')
        ax.text(R/2, -1.5, f'Range = {R:.1f} m', fontsize=10, ha='center')
        
        ax.set_xlabel('Horizontal Distance (m)')
        ax.set_ylabel('Height (m)')
        ax.set_title('Projectile Motion', fontweight='bold')
        ax.grid(True, alpha=0.3)
        return self._save_fig(fig)
    
    def _ray_diagram(self, p: Dict) -> BytesIO:
        fig, ax = plt.subplots(figsize=(10, 5))
        obj_dist = p.get('object_distance', 4)
        f = p.get('focal_length', 2)
        lens_type = p.get('type', 'convex')
        
        # Principal axis
        ax.axhline(y=0, color='black', lw=1)
        
        # Lens
        ax.annotate('', xy=(0, 1.8), xytext=(0, -1.8),
                   arrowprops=dict(arrowstyle='<->', color='blue', lw=2.5))
        ax.text(0.2, -0.3, 'O', fontsize=10)
        
        # Focal points
        ax.plot(-f, 0, 'ro', markersize=6)
        ax.plot(f, 0, 'ro', markersize=6)
        ax.text(-f, -0.4, 'F', fontsize=10, ha='center')
        ax.text(f, -0.4, "F'", fontsize=10, ha='center')
        
        # Object
        ax.annotate('', xy=(-obj_dist, 1), xytext=(-obj_dist, 0),
                   arrowprops=dict(arrowstyle='->', color='green', lw=2.5))
        ax.text(-obj_dist, 1.2, 'Object', fontsize=9, ha='center')
        
        # Calculate image (lens formula)
        u = -obj_dist
        v = (f * u) / (u - f)
        m = v / u
        img_h = m * 1
        
        # Rays
        ax.plot([-obj_dist, 0], [1, 1], 'r-', lw=1.5)  # Parallel ray
        ax.plot([0, v], [1, img_h], 'r-', lw=1.5)
        ax.plot([-obj_dist, v], [1, img_h], 'r--', lw=1.5, alpha=0.7)  # Through center
        
        # Image
        if v > 0:
            ax.annotate('', xy=(v, img_h), xytext=(v, 0),
                       arrowprops=dict(arrowstyle='->', color='purple', lw=2.5))
            ax.text(v, img_h-0.4, 'Image', fontsize=9, ha='center')
        
        ax.set_xlim(-obj_dist-1, max(6, v+2))
        ax.set_ylim(-2.5, 2.5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(f'{lens_type.capitalize()} Lens - Ray Diagram', fontweight='bold')
        return self._save_fig(fig)
    
    def _pulley(self, p: Dict) -> BytesIO:
        fig, ax = plt.subplots(figsize=(5, 7))
        m1 = p.get('m1', 5)
        m2 = p.get('m2', 3)
        
        # Support
        ax.plot([-0.5, 2], [5, 5], 'k-', lw=4)
        ax.fill_between([-0.5, 2], [5, 5], [5.1, 5.1], color='gray')
        
        # Pulley
        ax.add_patch(Circle((0.75, 4.6), 0.3, fill=False, edgecolor='black', lw=3))
        ax.plot([0.75, 0.75], [5, 4.9], 'k-', lw=2)
        
        # Left mass
        ax.plot([0.45, 0.45], [4.6, 2.2], 'k-', lw=2)
        ax.add_patch(Rectangle((0.1, 1.4), 0.7, 0.8, fill=True, facecolor='lightblue', edgecolor='black', lw=2))
        ax.text(0.45, 1.8, f'{m1}kg', fontsize=10, ha='center', va='center')
        
        # Right mass
        ax.plot([1.05, 1.05], [4.6, 3.0], 'k-', lw=2)
        ax.add_patch(Rectangle((0.7, 2.2), 0.7, 0.8, fill=True, facecolor='lightcoral', edgecolor='black', lw=2))
        ax.text(1.05, 2.6, f'{m2}kg', fontsize=10, ha='center', va='center')
        
        ax.set_xlim(-0.7, 2.2)
        ax.set_ylim(0.5, 5.5)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title('Pulley System', fontweight='bold')
        return self._save_fig(fig)
    
    def _spring(self, p: Dict) -> BytesIO:
        fig, ax = plt.subplots(figsize=(4, 7))
        k = p.get('k', 100)
        mass = p.get('mass', 2)
        ext = p.get('extension', 0.3)
        
        # Support
        ax.plot([-0.6, 0.6], [5, 5], 'k-', lw=4)
        ax.fill_between([-0.6, 0.6], [5, 5], [5.1, 5.1], color='gray')
        
        # Spring (zigzag)
        spring_top, spring_bot = 4.9, 2.5 - ext
        n_coils = 12
        ys = np.linspace(spring_top, spring_bot, n_coils*2+1)
        xs = [0] + [0.25 if i%2==1 else -0.25 for i in range(1, len(ys)-1)] + [0]
        ax.plot(xs, ys, 'b-', lw=2)
        
        # Mass
        ax.add_patch(Rectangle((-0.35, spring_bot-0.8), 0.7, 0.8, fill=True, 
                               facecolor='lightgreen', edgecolor='black', lw=2))
        ax.text(0, spring_bot-0.4, f'{mass}kg', fontsize=10, ha='center', va='center')
        
        # Labels
        ax.text(0.5, 3.7, f'k={k} N/m', fontsize=10)
        
        ax.set_xlim(-1, 1.2)
        ax.set_ylim(0.5, 5.5)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title('Spring-Mass System', fontweight='bold')
        return self._save_fig(fig)


class SVGGenerator:
    """Generates simple SVG illustrations via LLM"""
    
    def __init__(self, client, provider: str = "openai"):
        self.client = client
        self.provider = provider
        self.model = "gpt-4o" if provider == "openai" else "claude-sonnet-4-20250514"
    
    def generate(self, description: str) -> Optional[str]:
        """Generate SVG code from description"""
        prompt = f"""Generate simple SVG code for a physics diagram. Keep it minimal and clean.

Description: {description}

Requirements:
- Use only basic SVG elements: line, circle, rect, path, text, polygon
- Canvas size: width="300" height="200"
- Use black strokes (#000) and simple fills
- Include labels where appropriate
- Keep it simple - stick figures, basic shapes
- No external resources or complex effects

Return ONLY the SVG code, nothing else. Start with <svg and end with </svg>."""

        try:
            if self.provider == "anthropic":
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=1500,
                    messages=[{"role": "user", "content": prompt}]
                )
                svg = response.content[0].text
            else:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1500,
                    temperature=0.7
                )
                svg = response.choices[0].message.content
            
            # Clean up response - extract SVG
            svg = svg.strip()
            if '```' in svg:
                svg = re.search(r'<svg[^>]*>.*?</svg>', svg, re.DOTALL)
                svg = svg.group(0) if svg else None
            
            if svg and svg.startswith('<svg'):
                return svg
            return None
        except:
            return None


class TableParser:
    """Parses [TABLE: ...] format"""
    
    @staticmethod
    def parse(text: str) -> Optional[List[List[str]]]:
        match = re.search(r'\[TABLE:\s*([^\]]+)\]', text, re.IGNORECASE | re.DOTALL)
        if not match:
            return None
        
        content = match.group(1)
        rows = [r.strip() for r in content.split('|')]
        return [[c.strip() for c in row.split(',')] for row in rows if row]
    
    @staticmethod
    def to_html(data: List[List[str]]) -> str:
        if not data:
            return ""
        
        html = '<table style="border-collapse: collapse; width: 100%; margin: 1rem 0;">'
        # Header
        html += '<tr style="background-color: #1E88E5; color: white;">'
        for cell in data[0]:
            html += f'<th style="border: 1px solid #ddd; padding: 8px;">{cell}</th>'
        html += '</tr>'
        # Rows
        for i, row in enumerate(data[1:]):
            bg = '#f9f9f9' if i % 2 == 0 else '#fff'
            html += f'<tr style="background-color: {bg};">'
            for cell in row:
                html += f'<td style="border: 1px solid #ddd; padding: 8px; text-align: center;">{cell}</td>'
            html += '</tr>'
        html += '</table>'
        return html


# ============== VECTOR STORE ==============

class VectorStoreManager:
    def __init__(self, persist_dir: str, collection_name: str, api_key: str):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
            api_key=api_key, model_name="text-embedding-3-small"
        )
        try:
            self.collection = self.client.get_collection(name=collection_name, embedding_function=self.embedding_fn)
        except:
            self.collection = self.client.create_collection(name=collection_name, embedding_function=self.embedding_fn)
    
    def search(self, query: str, n_results: int = 5, doc_type: Optional[str] = None) -> List[Dict]:
        if self.collection.count() == 0:
            return []
        where = {"doc_type": doc_type} if doc_type else None
        results = self.collection.query(query_texts=[query], n_results=n_results, where=where)
        return [{"text": doc, "metadata": results['metadatas'][0][i]} 
                for i, doc in enumerate(results['documents'][0])] if results['documents'] else []
    
    def get_count(self) -> int:
        return self.collection.count()


# ============== QUESTION GENERATOR ==============

class QuestionGenerator:
    def __init__(self, api_key: str, provider: str = "openai"):
        self.provider = provider
        self.api_key = api_key
        
        if provider == "anthropic":
            from anthropic import Anthropic
            self.client = Anthropic(api_key=api_key)
            self.model = "claude-sonnet-4-20250514"
        else:
            self.client = OpenAI(api_key=api_key)
            self.model = "gpt-4o"
        
        self.svg_generator = SVGGenerator(self.client, provider)
    
    def _get_figure_instructions(self) -> str:
        return """
FIGURE INSTRUCTIONS - Include diagrams where appropriate using these formats:

1. TECHNICAL DIAGRAMS (will be auto-generated):
[FIGURE: circuit | resistors=2, voltage=12, series=true]
[FIGURE: wave | wavelength=2, amplitude=1, cycles=3]
[FIGURE: motion_graph | type=v-t, initial_velocity=0, acceleration=2]
[FIGURE: force_diagram | forces=weight,normal,friction]
[FIGURE: pendulum | length=2, angle=30]
[FIGURE: projectile | velocity=20, angle=45]
[FIGURE: ray_diagram | object_distance=4, focal_length=2, type=convex]
[FIGURE: pulley | m1=5, m2=3]
[FIGURE: spring | k=100, mass=2]

2. SIMPLE ILLUSTRATIONS (for scenarios):
[SVG: description of simple illustration needed]
Examples:
[SVG: A person pushing a box on a flat surface]
[SVG: A ball falling from a height with dotted path]
[SVG: Two cars moving in opposite directions]

3. DATA TABLES:
[TABLE: header1, header2, header3 | row1val1, row1val2, row1val3 | row2val1, row2val2, row2val3]
Example:
[TABLE: Time (s), Velocity (m/s), Distance (m) | 0, 0, 0 | 2, 10, 10 | 4, 20, 40]

Use at least 1-2 figures or tables per question set where relevant to the topic.
"""
    
    def generate_mcq(self, topic: str, context: str, num_questions: int = 5, 
                     difficulty: str = "medium", include_figures: bool = True) -> str:
        fig_inst = self._get_figure_instructions() if include_figures else ""
        
        prompt = f"""You are an expert Physics teacher creating questions for SSC (Class 9-10) students in Bangladesh following the NCTB curriculum.

Generate {num_questions} new MCQ questions on: "{topic}"

REFERENCE CONTENT:
{context}

{fig_inst}

REQUIREMENTS:
1. Each question has 4 options (ক, খ, গ, ঘ)
2. Match SSC Board exam style, difficulty: {difficulty}
3. Include correct answer and brief explanation
4. Test conceptual understanding
5. Include numerical problems where appropriate
6. Include at least 1-2 questions with figures/tables

FORMAT:
---
Question [number]:
[Question text with any [FIGURE: ...], [SVG: ...], or [TABLE: ...] as needed]

ক) [Option A]
খ) [Option B]
গ) [Option C]
ঘ) [Option D]

Correct Answer: [Letter]
Explanation: [Brief explanation]
---

Generate now:"""
        return self._call_llm(prompt)
    
    def generate_cq(self, topic: str, context: str, num_questions: int = 2,
                    difficulty: str = "medium", include_figures: bool = True) -> str:
        fig_inst = self._get_figure_instructions() if include_figures else ""
        
        prompt = f"""You are an expert Physics teacher creating CQ (Creative Questions) for SSC students in Bangladesh.

Generate {num_questions} new CQ on: "{topic}"

REFERENCE CONTENT:
{context}

{fig_inst}

CQ STRUCTURE:
- Stimulus (উদ্দীপক) with scenario/figure/table
- ক) Knowledge (1 mark)
- খ) Comprehension (2 marks)
- গ) Application with calculations (3 marks)
- ঘ) Higher-order analysis (4 marks)

REQUIREMENTS:
1. Each stimulus MUST include a [FIGURE: ...], [SVG: ...], or [TABLE: ...]
2. Make scenarios realistic and relatable
3. Difficulty: {difficulty}
4. Include calculations in part গ)

FORMAT:
---
Creative Question [number]:

Stimulus (উদ্দীপক):
[Scenario with [FIGURE: ...] or [SVG: ...] or [TABLE: ...]]

ক) [Knowledge question]
খ) [Comprehension question]
গ) [Application question]
ঘ) [Higher-order question]

Answer Key:
ক) [Answer]
খ) [Answer]
গ) [Step-by-step solution]
ঘ) [Detailed analysis]
---

Generate now:"""
        return self._call_llm(prompt)
    
    def _call_llm(self, prompt: str) -> str:
        try:
            if self.provider == "anthropic":
                response = self.client.messages.create(
                    model=self.model, max_tokens=4096,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text
            else:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=4096, temperature=0.7
                )
                return response.choices[0].message.content
        except Exception as e:
            return f"Error: {e}"


# ============== DISPLAY & PDF ==============

def process_and_display(content: str, fig_gen: MatplotlibFigureGenerator, svg_gen: SVGGenerator):
    """Process content and display with rendered figures/tables"""
    
    # Patterns
    fig_pattern = r'\[FIGURE:[^\]]+\]'
    svg_pattern = r'\[SVG:[^\]]+\]'
    table_pattern = r'\[TABLE:[^\]]+\]'
    
    # Split by patterns
    combined_pattern = f'({fig_pattern}|{svg_pattern}|{table_pattern})'
    parts = re.split(combined_pattern, content, flags=re.IGNORECASE)
    
    for part in parts:
        if not part or not part.strip():
            continue
        
        part_upper = part.upper()
        
        if '[FIGURE:' in part_upper:
            img_buf = fig_gen.generate(part)
            if img_buf:
                st.image(img_buf, use_container_width=True)
            else:
                st.code(part)
        
        elif '[SVG:' in part_upper:
            # Extract description
            match = re.search(r'\[SVG:\s*([^\]]+)\]', part, re.IGNORECASE)
            if match and svg_gen:
                desc = match.group(1)
                with st.spinner("Generating illustration..."):
                    svg_code = svg_gen.generate(desc)
                if svg_code:
                    st.markdown(f'<div class="figure-box">{svg_code}</div>', unsafe_allow_html=True)
                else:
                    st.info(f"📷 [Illustration: {desc}]")
            else:
                st.code(part)
        
        elif '[TABLE:' in part_upper:
            table_data = TableParser.parse(part)
            if table_data:
                st.markdown(TableParser.to_html(table_data), unsafe_allow_html=True)
            else:
                st.code(part)
        
        else:
            st.text(part)


def create_pdf(content: str, title: str, fig_gen: MatplotlibFigureGenerator) -> BytesIO:
    """Create PDF with figures and tables"""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=50, leftMargin=50, topMargin=50, bottomMargin=50)
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle('Title', parent=styles['Heading1'], fontSize=16, spaceAfter=20, alignment=1)
    body_style = ParagraphStyle('Body', parent=styles['Normal'], fontSize=10, leading=14, spaceAfter=8)
    
    story = [Paragraph(title, title_style), Spacer(1, 15)]
    
    # Process content
    fig_pattern = r'\[FIGURE:[^\]]+\]'
    table_pattern = r'\[TABLE:[^\]]+\]'
    svg_pattern = r'\[SVG:[^\]]+\]'
    combined = f'({fig_pattern}|{table_pattern}|{svg_pattern})'
    parts = re.split(combined, content, flags=re.IGNORECASE)
    
    for part in parts:
        if not part or not part.strip():
            continue
        
        part_upper = part.upper()
        
        if '[FIGURE:' in part_upper:
            img_buf = fig_gen.generate(part)
            if img_buf:
                story.append(RLImage(img_buf, width=4*inch, height=2.5*inch))
                story.append(Spacer(1, 10))
        
        elif '[TABLE:' in part_upper:
            table_data = TableParser.parse(part)
            if table_data:
                t = Table(table_data)
                t.setStyle(TableStyle([
                    ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1E88E5')),
                    ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
                    ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                    ('FONTSIZE', (0,0), (-1,-1), 9),
                    ('GRID', (0,0), (-1,-1), 1, colors.black),
                ]))
                story.append(t)
                story.append(Spacer(1, 10))
        
        elif '[SVG:' in part_upper:
            # SVG can't be easily added to ReportLab, add placeholder
            match = re.search(r'\[SVG:\s*([^\]]+)\]', part, re.IGNORECASE)
            if match:
                story.append(Paragraph(f"[Illustration: {match.group(1)}]", body_style))
        
        else:
            for line in part.split('\n'):
                if line.strip():
                    safe = line.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                    story.append(Paragraph(safe, body_style))
    
    doc.build(story)
    buffer.seek(0)
    return buffer


# ============== MAIN APP ==============

def main():
    st.markdown('<p class="main-header">📚 NCTB Question Generator</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Generate MCQ and Creative Questions with Diagrams</p>', unsafe_allow_html=True)
    
    # Initialize generators
    fig_gen = MatplotlibFigureGenerator()
    
    col_settings, col_output = st.columns([1, 1.5])
    
    with col_settings:
        st.markdown("### ⚙️ Settings")
        
        with st.expander("🔑 API Configuration", expanded=True):
            api_provider = st.selectbox("AI Provider", ["OpenAI (GPT-4)", "Anthropic (Claude)"], index=0)
            api_key = st.text_input("API Key", type="password", placeholder="sk-... or sk-ant-...")
            if api_key:
                st.success("✓ API Key entered")
            else:
                st.warning("Please enter your API key")
        
        st.markdown("---")
        st.markdown("### 📖 Curriculum")
        
        selected_class = st.selectbox("Class", list(CURRICULUM_DATA.keys()))
        selected_subject = st.selectbox("Subject", list(CURRICULUM_DATA[selected_class].keys()))
        chapters = CURRICULUM_DATA[selected_class][selected_subject]["chapters"]
        selected_chapters = st.multiselect("Chapter(s)", chapters, default=[chapters[0]] if chapters else [])
        
        st.markdown("---")
        st.markdown("### 📝 Question Settings")
        
        question_type = st.selectbox("Type", ["MCQ (Multiple Choice)", "CQ (Creative Questions)", "Both"])
        
        col_n, col_d = st.columns(2)
        with col_n:
            if question_type == "MCQ (Multiple Choice)":
                num_q = st.number_input("MCQs", 1, 20, 5)
            elif question_type == "CQ (Creative Questions)":
                num_q = st.number_input("CQs", 1, 10, 2)
            else:
                num_mcq = st.number_input("MCQs", 1, 20, 5)
                num_cq = st.number_input("CQs", 1, 10, 2)
        
        with col_d:
            difficulty = st.selectbox("Difficulty", ["Easy", "Medium", "Hard"], index=1)
        
        include_figures = st.checkbox("Include Figures & Tables", value=True)
        
        st.markdown("---")
        generate_btn = st.button("🚀 Generate Questions", type="primary", use_container_width=True)
    
    with col_output:
        st.markdown("### 📄 Generated Questions")
        
        if generate_btn:
            if not api_key:
                st.error("❌ Please enter API key!")
            elif not selected_chapters:
                st.error("❌ Select at least one chapter!")
            else:
                provider = "anthropic" if "Anthropic" in api_provider else "openai"
                
                with st.spinner("🔄 Generating questions..."):
                    try:
                        # Vector store
                        vs = VectorStoreManager("chroma_db", "physics_nctb", api_key)
                        if vs.get_count() == 0:
                            st.warning("⚠️ No content. Run `python main.py --setup` first.")
                            st.stop()
                        
                        topics = " ".join([c.split(": ")[-1] for c in selected_chapters])
                        
                        tb_results = vs.search(topics, 5, "textbook")
                        q_results = vs.search(topics, 3, "past_question")
                        
                        context = ""
                        if tb_results:
                            context += "=== TEXTBOOK ===\n" + "\n".join([r["text"] for r in tb_results])
                        if q_results:
                            context += "\n=== PAST QUESTIONS ===\n" + "\n".join([r["text"] for r in q_results])
                        
                        if not context:
                            st.warning("⚠️ No content found.")
                            st.stop()
                        
                        gen = QuestionGenerator(api_key, provider)
                        
                        if question_type == "MCQ (Multiple Choice)":
                            result = gen.generate_mcq(topics, context, num_q, difficulty.lower(), include_figures)
                        elif question_type == "CQ (Creative Questions)":
                            result = gen.generate_cq(topics, context, num_q, difficulty.lower(), include_figures)
                        else:
                            mcq = gen.generate_mcq(topics, context, num_mcq, difficulty.lower(), include_figures)
                            cq = gen.generate_cq(topics, context, num_cq, difficulty.lower(), include_figures)
                            result = f"=== MCQ ===\n\n{mcq}\n\n=== CQ ===\n\n{cq}"
                        
                        st.session_state['questions'] = result
                        st.session_state['title'] = f"{selected_subject} - {', '.join([c.split(': ')[-1] for c in selected_chapters])}"
                        st.session_state['api_key'] = api_key
                        st.session_state['provider'] = provider
                        
                        st.success("✅ Generated!")
                    
                    except Exception as e:
                        st.error(f"❌ Error: {e}")
        
        # Display
        if 'questions' in st.session_state and st.session_state['questions']:
            # Create SVG generator if we have API key
            svg_gen = None
            if 'api_key' in st.session_state:
                provider = st.session_state.get('provider', 'openai')
                if provider == "anthropic":
                    from anthropic import Anthropic
                    client = Anthropic(api_key=st.session_state['api_key'])
                else:
                    client = OpenAI(api_key=st.session_state['api_key'])
                svg_gen = SVGGenerator(client, provider)
            
            process_and_display(st.session_state['questions'], fig_gen, svg_gen)
            
            st.markdown("---")
            c1, c2 = st.columns(2)
            
            with c1:
                st.download_button("📥 Download TXT", st.session_state['questions'], 
                                   "questions.txt", "text/plain", use_container_width=True)
            
            with c2:
                try:
                    pdf = create_pdf(st.session_state['questions'], 
                                    st.session_state.get('title', 'Questions'), fig_gen)
                    st.download_button("📥 Download PDF", pdf, "questions.pdf", 
                                      "application/pdf", use_container_width=True)
                except Exception as e:
                    st.warning(f"PDF failed: {e}")
        else:
            st.info("👈 Configure and click Generate!")
    
    st.markdown("---")
    st.markdown("<p style='text-align: center; color: #888;'>NCTB Curriculum | Bangladesh</p>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
