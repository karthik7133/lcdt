from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                 TableStyle, HRFlowable, KeepTogether)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
import numpy as np

# ── Color palette ──────────────────────────────────────────────
NAVY        = colors.HexColor('#0D1B2A')
CYAN_DARK   = colors.HexColor('#007A8C')
CYAN_BG     = colors.HexColor('#E0F4F7')
CYAN_MID    = colors.HexColor('#00B8D4')
MAGENTA     = colors.HexColor('#8B005D')
MAGENTA_BG  = colors.HexColor('#FCE4F5')
GREEN_DARK  = colors.HexColor('#1B6B2F')
GREEN_BG    = colors.HexColor('#E6F5EA')
RED_DARK    = colors.HexColor('#8B0000')
RED_BG      = colors.HexColor('#FDECEA')
GREY_LIGHT  = colors.HexColor('#F5F5F5')
GREY_MED    = colors.HexColor('#CCCCCC')
GREY_DARK   = colors.HexColor('#555555')
BLACK       = colors.HexColor('#111111')
WHITE       = colors.white
GOLD        = colors.HexColor('#B8860B')
GOLD_BG     = colors.HexColor('#FFF8E1')
PURPLE      = colors.HexColor('#4A0072')
PURPLE_BG   = colors.HexColor('#F3E5FF')

W, H = A4

doc = SimpleDocTemplate(
    'Cyber_Watchdog_Algorithm_Structure.pdf',
    pagesize=A4,
    leftMargin=18*mm, rightMargin=18*mm,
    topMargin=16*mm, bottomMargin=16*mm,
)

SS = getSampleStyleSheet()

def sty(name='Normal', **kw):
    base = SS[name]
    return ParagraphStyle(name+'_custom', parent=base, **kw)

title_sty   = sty('Normal', fontSize=18, leading=22, textColor=WHITE,
                   fontName='Helvetica-Bold', alignment=TA_CENTER)
sub_sty     = sty('Normal', fontSize=10, leading=14, textColor=colors.HexColor('#CCDDFF'),
                   fontName='Helvetica', alignment=TA_CENTER)
h1_sty      = sty('Normal', fontSize=13, leading=17, textColor=WHITE,
                   fontName='Helvetica-Bold', spaceAfter=2)
h2_sty      = sty('Normal', fontSize=11, leading=15, textColor=NAVY,
                   fontName='Helvetica-Bold', spaceBefore=8, spaceAfter=4)
body_sty    = sty('Normal', fontSize=9, leading=13, textColor=BLACK,
                   fontName='Helvetica')
mono_sty    = sty('Normal', fontSize=8, leading=12, textColor=colors.HexColor('#1A1A2E'),
                   fontName='Courier', backColor=colors.HexColor('#F0F0F8'))
eq_sty      = sty('Normal', fontSize=10, leading=14, textColor=NAVY,
                   fontName='Courier-Bold', alignment=TA_CENTER)

def sp(h=4): return Spacer(1, h)

def section_header(text, bg=NAVY, fg=WHITE):
    tbl = Table([[Paragraph(text, sty('Normal', fontSize=12, leading=16,
                  textColor=fg, fontName='Helvetica-Bold'))]], colWidths=[W - 36*mm])
    tbl.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), bg),
        ('ROUNDEDCORNERS', [4,4,4,4]),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('LEFTPADDING', (0,0), (-1,-1), 12),
    ]))
    return tbl

story = []

# ══════════════════════════════════════════════════════════════
# TITLE BLOCK
# ══════════════════════════════════════════════════════════════
title_tbl = Table([[
    Paragraph('Cyber Watchdog: Algorithm & Architecture Structure', title_sty),
    ],[
    Paragraph('Multi-Timescale Dynamical Systems (CRT Model) & Logistic Regression', sub_sty),
    ],[
    Paragraph('Official Technical Documentation of the Risk Inference Engine', sub_sty),
]], colWidths=[W - 36*mm])
title_tbl.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,-1), NAVY),
    ('ALIGN', (0,0), (-1,-1), 'CENTER'),
    ('TOPPADDING', (0,0), (-1,-1), 15),
    ('BOTTOMPADDING', (0,0), (-1,-1), 15),
]))
story.append(title_tbl)
story.append(sp(10))

# ══════════════════════════════════════════════════════════════
# 1. OVERVIEW
# ══════════════════════════════════════════════════════════════
story.append(section_header('1. Architectural Strategy: The 5-Layer Digital Twin'))
story.append(sp(5))
story.append(Paragraph(
    'The Cyber Watchdog implements a layered architecture to transform raw telemetry into predictive security insights. '
    'It models the human user as a <b>Dynamical System</b> where internal states (Stress, Energy, Habits) evolve over time '
    'driven by external events and biological constraints.', body_sty))
story.append(sp(8))

# ══════════════════════════════════════════════════════════════
# LAYER 1: OBSERVATION SPACE
# ══════════════════════════════════════════════════════════════
story.append(section_header('Layer 1: Multimodal Observation Space (X<sub>t</sub>)', bg=CYAN_DARK))
story.append(sp(5))
story.append(Paragraph(
    'Every 10-second interval, the system captures a vector X<sub>t</sub> consisting of 12 primary signals. '
    'These signals serve as the raw inputs for the state inference engine.', body_sty))
story.append(sp(6))

l1_data = [
    [Paragraph('<b>Sensor Family</b>', sty('Normal', fontSize=9, textColor=WHITE, fontName='Helvetica-Bold')),
     Paragraph('<b>Signal Parameter</b>', sty('Normal', fontSize=9, textColor=WHITE, fontName='Helvetica-Bold')),
     Paragraph('<b>Description & Impact</b>', sty('Normal', fontSize=9, textColor=WHITE, fontName='Helvetica-Bold'))],
    
    [Paragraph('<b>Work Context</b>', body_sty), Paragraph('1. Keystrokes', body_sty), Paragraph('Total count of keys pressed. Indicates active workload.', body_sty)],
    [Paragraph('', body_sty), Paragraph('2. Mouse Entropy', body_sty), Paragraph('Complexity of movement. High entropy = intense activity.', body_sty)],
    [Paragraph('', body_sty), Paragraph('3. Typing Errors', body_sty), Paragraph('Backspaces/Deletes. High rate = cognitive friction/exhaustion.', body_sty)],
    [Paragraph('', body_sty), Paragraph('4. Task Switching', body_sty), Paragraph('Window changes. High frequency = context-switching stress.', body_sty)],
    
    [Paragraph('<b>Environment</b>', body_sty), Paragraph('5. Notifications', body_sty), Paragraph('Audio system alerts. Represents external interruptions.', body_sty)],
    [Paragraph('', body_sty), Paragraph('6. Email Patterns', body_sty), Paragraph('Incoming frequency & unknown senders. High volume = pressure.', body_sty)],
    
    [Paragraph('<b>Digital Habits</b>', body_sty), Paragraph('7. Password Habits', body_sty), Paragraph('Manager use (Good) vs Manual (Risk). Affects Habit state H<sub>t</sub>.', body_sty)],
    [Paragraph('', body_sty), Paragraph('8. Browser Exposure', body_sty), Paragraph('HTTP sites/Webmail. Direct indicator of security hygiene.', body_sty)],
    [Paragraph('', body_sty), Paragraph('9. OS Update Behavior', body_sty), Paragraph('Delayed patches. Represents neglect of system health.', body_sty)],
    
    [Paragraph('<b>Biomarkers</b>', body_sty), Paragraph('10. Sleep Deficit', body_sty), Paragraph('Working during rest hours. Reduces Capacity baseline &mu;.', body_sty)],
    [Paragraph('', body_sty), Paragraph('11. Phishing Sims', body_sty), Paragraph('Interaction with bait links. Spikes Adversarial state A<sub>t</sub>.', body_sty)],
    [Paragraph('', body_sty), Paragraph('12. Vision Fatigue', body_sty), Paragraph('Blinks/Slumping (AI Camera). Rapidly drains Capacity C<sub>t</sub>.', body_sty)],
]

l1_tbl = Table(l1_data, colWidths=[(W-36*mm)*f for f in [0.2, 0.25, 0.55]])
l1_tbl.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,0), CYAN_DARK),
    ('GRID', (0,0), (-1,-1), 0.5, GREY_MED),
    ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ('TOPPADDING', (0,0), (-1,-1), 6),
    ('BOTTOMPADDING', (0,0), (-1,-1), 6),
    ('LEFTPADDING', (0,0), (-1,-1), 8),
]))
story.append(l1_tbl)
story.append(sp(10))

# ══════════════════════════════════════════════════════════════
# LAYER 2 & 3: CRT MODEL
# ══════════════════════════════════════════════════════════════
story.append(section_header('Layer 2 & 3: The CRT (Capacity, Reserve, Threat) Model', bg=PURPLE))
story.append(sp(5))
story.append(Paragraph(
    'The system maps X<sub>t</sub> into four hidden latent states Z<sub>t</sub>. These states evolve using '
    '<b>Multi-Timescale Dynamics</b>, ensuring that risk predictions have temporal momentum.', body_sty))
story.append(sp(6))

# Formula Block
def eq_box(title, formula, bg=PURPLE_BG, border=PURPLE):
    content = [
        Paragraph(f'<b>{title}</b>', sty('Normal', fontSize=10, textColor=NAVY, fontName='Helvetica-Bold')),
        sp(2),
        Table([[Paragraph(formula, eq_sty)]], colWidths=[W-36*mm], style=[('BACKGROUND', (0,0), (-1,-1), bg), ('BOX', (0,0), (-1,-1), 1, border), ('TOPPADDING', (0,0), (-1,-1), 8), ('BOTTOMPADDING', (0,0), (-1,-1), 8)])
    ]
    return KeepTogether(content)

# Capacity
story.append(eq_box('A. Slow Dynamics: Cognitive Capacity (C<sub>t</sub>)', 
                    'dC/dt = &alpha; (&mu; - C<sub>t</sub>)'))
story.append(Paragraph(
    '<b>Logic:</b> C<sub>t</sub> is your "energy tank." It drifts toward baseline &mu;. '
    'If <b>Sleep Deficit</b> or <b>Vision Fatigue</b> are detected, &mu; drops. '
    'Recovery speed &alpha; is much slower than drain speed.', body_sty))
story.append(sp(6))

# Demand
story.append(eq_box('B. Fast Dynamics: Cognitive Demand (D<sub>t</sub>)', 
                    'dD/dt = f(work) - &beta; (D<sub>t</sub> - 0.1)'))
story.append(Paragraph(
    '<b>Logic:</b> D<sub>t</sub> is current stress. <b>f(work)</b> is a weighted sum of <b>Keystrokes, Switches, '
    'Errors, and Notifications</b>. &beta; is the "cooling rate" that pull stress back to idle (0.1) over time.', body_sty))
story.append(sp(6))

# Habits
story.append(eq_box('C. Medium Dynamics: Habit State (H<sub>t</sub>)', 
                    'H<sub>t+1</sub> = H<sub>t</sub> + &eta; * Reward'))
story.append(Paragraph(
    '<b>Logic:</b> Security hygiene. <b>Good Password Habits</b> give a positive reward; '
    '<b>Insecure Browser Exposure</b> or <b>Delayed Updates</b> give a penalty. '
    '&eta; determines how fast the user "learns" or "forgets" good habits.', body_sty))
story.append(sp(6))

# Reserve Gap
story.append(eq_box('D. Core Output: Cognitive Reserve Gap (CRG<sub>t</sub>)', 
                    'CRG<sub>t</sub> = C<sub>t</sub> - D<sub>t</sub>'))
story.append(Paragraph(
    '<b>Logic:</b> The most critical safety indicator. <b>Positive CRG</b> = spare brainpower. '
    '<b>Negative CRG</b> = Cognitive Debt (extremely high risk of error).', body_sty))
story.append(sp(10))

# ══════════════════════════════════════════════════════════════
# LAYER 4: RISK CALCULATION
# ══════════════════════════════════════════════════════════════
story.append(section_header('Layer 4: Risk Forecasting Engine (Logistic Regression)', bg=RED_DARK))
story.append(sp(5))
story.append(Paragraph(
    'To convert complex hidden states into a single, actionable percentage, we use '
    '<b>Logistic Regression</b> with calibrated weights.', body_sty))
story.append(sp(6))

# Step 1
story.append(Paragraph('<b>Step 1: The Linear Logit (z)</b>', h2_sty))
story.append(Table([[Paragraph('z = &beta;<sub>1</sub> * CRG<sub>t</sub> + &beta;<sub>2</sub> * H<sub>t</sub> + &beta;<sub>3</sub> * A<sub>t</sub> + bias', eq_sty)]], 
                    colWidths=[W-36*mm], style=[('BACKGROUND', (0,0), (-1,-1), RED_BG), ('TOPPADDING', (0,0), (-1,-1), 10), ('BOTTOMPADDING', (0,0), (-1,-1), 10)]))
story.append(sp(4))

weights_data = [
    [Paragraph('<b>Weight</b>', body_sty), Paragraph('<b>Value</b>', body_sty), Paragraph('<b>Reasoning</b>', body_sty)],
    [Paragraph('&beta;<sub>1</sub> (Reserve)', body_sty), Paragraph('-3.0', body_sty), Paragraph('High reserve (spare capacity) reduces risk linearly.', body_sty)],
    [Paragraph('&beta;<sub>2</sub> (Habits)', body_sty), Paragraph('-3.0', body_sty), Paragraph('Good habits act as a protective "shield" reducing risk.', body_sty)],
    [Paragraph('&beta;<sub>3</sub> (Threat)', body_sty), Paragraph('+6.0', body_sty), Paragraph('Active phishing click is a catastrophic risk multiplier.', body_sty)],
    [Paragraph('Bias', body_sty), Paragraph('+0.5', body_sty), Paragraph('Base risk level assuming standard conditions.', body_sty)],
]
w_tbl = Table(weights_data, colWidths=[(W-36*mm)*0.2, (W-36*mm)*0.15, (W-36*mm)*0.65])
w_tbl.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 0.4, GREY_MED), ('BACKGROUND', (0,0), (-1,0), GREY_LIGHT)]))
story.append(w_tbl)
story.append(sp(8))

# Step 2
story.append(Paragraph('<b>Step 2: The Sigmoid "Squash"</b>', h2_sty))
story.append(Paragraph(
    'We use the Sigmoid function to transform the Z-score into a probability P(M<sub>t</sub>) in the range [0.0, 1.0].', body_sty))
story.append(sp(4))
story.append(Table([[Paragraph('P(Mistake) = 1 / (1 + e<super>-z</super>)', eq_sty)]], 
                    colWidths=[W-36*mm], style=[('BACKGROUND', (0,0), (-1,-1), RED_BG), ('BOX', (0,0), (-1,-1), 1.5, RED_DARK), ('TOPPADDING', (0,0), (-1,-1), 12), ('BOTTOMPADDING', (0,0), (-1,-1), 12)]))
story.append(sp(10))

# ══════════════════════════════════════════════════════════════
# LAYER 5 & ARCHITECTURE
# ══════════════════════════════════════════════════════════════
story.append(section_header('Layer 5: Digital Twin Simulation & Architecture', bg=colors.HexColor('#1A3A5C')))
story.append(sp(5))
story.append(Paragraph(
    'Layer 5 uses the underlying CRT model to run <b>Counterfactual Simulations</b>. It projects the trajectory '
    'of the user state 30 days into the future to identify burnout points before they happen.', body_sty))
story.append(sp(8))

# Architecture mapping
story.append(Paragraph('<b>Unified Architecture Linkage:</b>', h2_sty))
arch_data = [
    [Paragraph('<b>Layer</b>', body_sty), Paragraph('<b>Function</b>', body_sty), Paragraph('<b>Algorithm Component</b>', body_sty)],
    [Paragraph('L1: Telemetry', body_sty), Paragraph('Data Collection', body_sty), Paragraph('The 12-Signal Vector X<sub>t</sub>')],
    [Paragraph('L2/3: Inference', body_sty), Paragraph('Hidden State Modeling', body_sty), Paragraph('CRT Dynamical Systems (Euler Method)')],
    [Paragraph('L4: Forecasting', body_sty), Paragraph('Instant Risk Score', body_sty), Paragraph('Logistic Regression + Sigmoid')],
    [Paragraph('L5: Simulation', body_sty), Paragraph('Predictive Trajectory', body_sty), Paragraph('Heuristic Burnout Projections')],
]
a_tbl = Table(arch_data, colWidths=[(W-36*mm)*0.25, (W-36*mm)*0.3, (W-36*mm)*0.45])
a_tbl.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 0.4, GREY_MED), ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#E5EDF5'))]))
story.append(a_tbl)
story.append(sp(15))

# Footer
story.append(HRFlowable(width="100%", thickness=1, color=NAVY))
story.append(sp(2))
story.append(Paragraph('Cyber Watchdog Technical Documentation * Digital Twin Framework * v5.0', sty('Normal', fontSize=7, textColor=GREY_DARK, alignment=TA_CENTER)))

doc.build(story)
print("PDF Generated: Cyber_Watchdog_Algorithm_Structure.pdf")
