from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                 TableStyle, HRFlowable, KeepTogether)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.platypus import PageBreak

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
    'LCDT_Comprehensive_Manual.pdf',
    pagesize=A4,
    leftMargin=18*mm, rightMargin=18*mm,
    topMargin=16*mm, bottomMargin=16*mm,
)

SS = getSampleStyleSheet()

def sty(name='Normal', **kw):
    base = SS[name]
    return ParagraphStyle(name+'_custom', parent=base, **kw)

title_sty   = sty('Normal', fontSize=15, leading=20, textColor=WHITE,
                   fontName='Helvetica-Bold', alignment=TA_CENTER)
sub_sty     = sty('Normal', fontSize=9, leading=13, textColor=colors.HexColor('#CCDDFF'),
                   fontName='Helvetica', alignment=TA_CENTER)
h1_sty      = sty('Normal', fontSize=12, leading=16, textColor=WHITE,
                   fontName='Helvetica-Bold', spaceAfter=2)
h2_sty      = sty('Normal', fontSize=10, leading=14, textColor=NAVY,
                   fontName='Helvetica-Bold', spaceBefore=6, spaceAfter=3)
body_sty    = sty('Normal', fontSize=8.5, leading=13, textColor=BLACK,
                   fontName='Helvetica')
mono_sty    = sty('Normal', fontSize=7.8, leading=12, textColor=colors.HexColor('#1A1A2E'),
                   fontName='Courier', backColor=colors.HexColor('#F0F0F8'))
label_sty   = sty('Normal', fontSize=8, leading=11, textColor=GREY_DARK,
                   fontName='Helvetica')
eq_sty      = sty('Normal', fontSize=9, leading=13, textColor=NAVY,
                   fontName='Courier-Bold', alignment=TA_CENTER)
small_sty   = sty('Normal', fontSize=7.5, leading=11, textColor=GREY_DARK,
                   fontName='Helvetica')

def sp(h=4): return Spacer(1, h)
def hr(c=GREY_MED): return HRFlowable(width='100%', thickness=0.5, color=c, spaceAfter=4, spaceBefore=4)

def section_header(text, bg=NAVY, fg=WHITE):
    tbl = Table([[Paragraph(text, sty('Normal', fontSize=11, leading=15,
                  textColor=fg, fontName='Helvetica-Bold'))]], colWidths=[W - 36*mm])
    tbl.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), bg),
        ('ROUNDEDCORNERS', [4,4,4,4]),
        ('TOPPADDING', (0,0), (-1,-1), 7),
        ('BOTTOMPADDING', (0,0), (-1,-1), 7),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
    ]))
    return tbl

def code_block(lines):
    joined = '<br/>'.join(lines)
    tbl = Table([[Paragraph(joined, sty('Normal', fontSize=7.5, leading=12,
                  textColor=colors.HexColor('#1A1A2E'), fontName='Courier'))]], 
                colWidths=[W - 36*mm])
    tbl.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#EEF0FF')),
        ('BOX', (0,0), (-1,-1), 0.8, colors.HexColor('#AAAACC')),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('RIGHTPADDING', (0,0), (-1,-1), 10),
    ]))
    return tbl

story = []

# ══════════════════════════════════════════════════════════════
# TITLE BLOCK
# ══════════════════════════════════════════════════════════════
title_tbl = Table([[
    Paragraph('Lifelong Cognitive–Cyber Digital Twin', title_sty),
    ],[
    Paragraph('Notification Audio Monitor — LCDT Framework Mapping', sub_sty),
    ],[
    Paragraph('Workflow Problem Solved Using Official LCDT Symbols &amp; Formulas', sub_sty),
]], colWidths=[W - 36*mm])
title_tbl.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,-1), NAVY),
    ('ALIGN', (0,0), (-1,-1), 'CENTER'),
    ('TOPPADDING', (0,0), (-1,-1), 10),
    ('BOTTOMPADDING', (0,0), (-1,-1), 10),
    ('LEFTPADDING', (0,0), (-1,-1), 12),
    ('RIGHTPADDING', (0,0), (-1,-1), 12),
]))
story.append(title_tbl)
story.append(sp(8))

# ══════════════════════════════════════════════════════════════
# 1. ARCHITECTURAL OVERVIEW
# ══════════════════════════════════════════════════════════════
story.append(section_header('1. Architectural Overview'))
story.append(sp(5))
story.append(Paragraph(
    'The Cyber Watchdog implements a 5-layer Digital Twin architecture that models the human internal state '
    'as a dynamical system. This document outlines the mathematical mapping from raw sensor observations '
    'to predictive risk outcomes.', body_sty))
story.append(sp(10))

# ══════════════════════════════════════════════════════════════
# LAYER 1
# ══════════════════════════════════════════════════════════════
story.append(section_header('Layer 1 — Human Sensing: Multimodal Observation Space X<sub>t</sub>', bg=CYAN_DARK))
story.append(sp(5))
story.append(Paragraph(
    'Every 10 seconds the system gathers the raw signal vector. The LCDT notation defines this as the '
    'Observation X<sub>t</sub> containing 12 parameters across four signal families:',
    body_sty))
story.append(sp(6))

col_w = [(W-36*mm)*f for f in [0.22, 0.35, 0.22, 0.21]]
hdr = [
    Paragraph('<b>Signal Family</b>', sty('Normal', fontSize=8, textColor=WHITE, fontName='Helvetica-Bold')),
    Paragraph('<b>Parameter in X<sub>t</sub></b>', sty('Normal', fontSize=8, textColor=WHITE, fontName='Helvetica-Bold')),
    Paragraph('<b>Our Monitor\'s Variable</b>', sty('Normal', fontSize=8, textColor=WHITE, fontName='Helvetica-Bold')),
    Paragraph('<b>Status</b>', sty('Normal', fontSize=8, textColor=WHITE, fontName='Helvetica-Bold', alignment=TA_CENTER)),
]

def cp(t, **kw): return Paragraph(t, sty('Normal', fontSize=8, leading=12, fontName='Helvetica', **kw))

rows_l1 = [
    hdr,
    [cp('Adversarial Exposure'), cp('Phishing simulations, Scam tasks'),
     cp('peak > 0.05 threshold'), cp('✓ Mapped', alignment=TA_CENTER, textColor=GREEN_DARK)],
    [cp('Digital Behaviour'), cp('Password habits, Browser exposure'),
     cp('target_processes list'), cp('✓ Mapped', alignment=TA_CENTER, textColor=GREEN_DARK)],
    [cp('Work Context'), cp('Notifications'),
     cp('notification_count'), cp('✓ Mapped', alignment=TA_CENTER, textColor=GREEN_DARK)],
    [cp('Cognitive Biomarkers'), cp('Task switching, Typing errors'),
     cp('Not in this module'), cp('– Partial', alignment=TA_CENTER, textColor=GREY_DARK)],
]

l1_tbl = Table(rows_l1, colWidths=col_w)
l1_tbl.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,0), CYAN_DARK),
    ('BACKGROUND', (0,1), (-1,1), GREY_LIGHT),
    ('BACKGROUND', (0,2), (-1,2), WHITE),
    ('BACKGROUND', (0,3), (-1,3), CYAN_BG),
    ('BACKGROUND', (0,4), (-1,4), GREY_LIGHT),
    ('ROWBACKGROUNDS', (0,1), (-1,-1), [GREY_LIGHT, WHITE]),
    ('GRID', (0,0), (-1,-1), 0.4, GREY_MED),
    ('TOPPADDING', (0,0), (-1,-1), 6),
    ('BOTTOMPADDING', (0,0), (-1,-1), 6),
    ('LEFTPADDING', (0,0), (-1,-1), 7),
    ('RIGHTPADDING', (0,0), (-1,-1), 7),
    ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
]))
story.append(l1_tbl)
story.append(sp(10))

# ══════════════════════════════════════════════════════════════
# LAYER 2
# ══════════════════════════════════════════════════════════════
story.append(section_header('Layer 2 — Latent Cognitive–Cyber State Inference', bg=PURPLE))
story.append(sp(5))
story.append(Paragraph(
    'Layer 2 uses a Bayesian Deep State-Space Model to map the raw observations X<sub>t</sub> into the four '
    'Hidden Roots of the internal state vector:', body_sty))
story.append(sp(5))

eq_tbl = Table([[Paragraph(
    'Z<sub>t</sub> = [ C<sub>t</sub> , D<sub>t</sub> , H<sub>t</sub> , A<sub>t</sub> ]',
    sty('Normal', fontSize=13, leading=18, textColor=NAVY, fontName='Courier-Bold', alignment=TA_CENTER)
)]], colWidths=[W-36*mm])
eq_tbl.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,-1), PURPLE_BG),
    ('BOX', (0,0), (-1,-1), 1, PURPLE),
    ('TOPPADDING', (0,0), (-1,-1), 8),
    ('BOTTOMPADDING', (0,0), (-1,-1), 8),
]))
story.append(eq_tbl)
story.append(sp(6))

# Layer 2 state table
col_w2 = [(W-36*mm)*f for f in [0.07, 0.22, 0.35, 0.24, 0.12]]
def mkc(t, color=BLACK, bold=False, align=TA_LEFT):
    fn = 'Helvetica-Bold' if bold else 'Helvetica'
    return Paragraph(t, sty('Normal', fontSize=8, leading=12, textColor=color, fontName=fn, alignment=align))

hdr2 = [mkc('Symbol', WHITE, True), mkc('Name', WHITE, True), mkc('What it Means', WHITE, True),
        mkc('Driven by', WHITE, True), mkc('Range', WHITE, True, TA_CENTER)]
rows_l2 = [
    hdr2,
    [mkc('C<sub>t</sub>', CYAN_DARK, True),
     mkc('Cognitive Capacity (Cyan)', CYAN_DARK, True),
     mkc('Mental energy available — reduced by fatigue &amp; sleep deficit'),
     mkc('Vision Fatigue, Sleep Deficit, Late-night work'),
     mkc('0.0 → 1.0', align=TA_CENTER)],
    [mkc('D<sub>t</sub>', MAGENTA, True),
     mkc('Cognitive Demand (Magenta)', MAGENTA, True),
     mkc('Stress load — raised by interruptions &amp; task-switching'),
     mkc('Task Switches, Notifications ← our sensor'),
     mkc('0.1 → 1.0', align=TA_CENTER)],
    [mkc('H<sub>t</sub>', GREEN_DARK, True),
     mkc('Habit State (Green)', GREEN_DARK, True),
     mkc('Quality of learned security behaviour'),
     mkc('Password manager use, Browser exposure'),
     mkc('0.0 → 1.0', align=TA_CENTER)],
    [mkc('A<sub>t</sub>', RED_DARK, True),
     mkc('Adversarial Exposure (Red)', RED_DARK, True),
     mkc('Active threat level — spikes on phishing events'),
     mkc('Phishing simulations, Scam task interactions'),
     mkc('0.0 → 1.0', align=TA_CENTER)],
]
l2_tbl = Table(rows_l2, colWidths=col_w2)
l2_tbl.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,0), PURPLE),
    ('BACKGROUND', (0,1), (-1,1), CYAN_BG),
    ('BACKGROUND', (0,2), (-1,2), MAGENTA_BG),
    ('BACKGROUND', (0,3), (-1,3), GREEN_BG),
    ('BACKGROUND', (0,4), (-1,4), RED_BG),
    ('GRID', (0,0), (-1,-1), 0.4, GREY_MED),
    ('TOPPADDING', (0,0), (-1,-1), 6),
    ('BOTTOMPADDING', (0,0), (-1,-1), 6),
    ('LEFTPADDING', (0,0), (-1,-1), 7),
    ('RIGHTPADDING', (0,0), (-1,-1), 7),
    ('VALIGN', (0,0), (-1,-1), 'TOP'),
]))
story.append(l2_tbl)
story.append(sp(5))
story.append(Paragraph(
    'The output of Layer 2 is the posterior distribution <b>P(Z<sub>t</sub> | X<sub>1:t</sub>)</b> — the Digital '
    'Twin\'s current internal state given all observations up to time t. Our notification count feeds directly into '
    'D<sub>t</sub> (Demand) via the Work Pressure sub-calculation.', body_sty))
story.append(sp(10))

# ══════════════════════════════════════════════════════════════
# LAYER 3
# ══════════════════════════════════════════════════════════════
story.append(section_header('Layer 3 — Multi-Timescale Dynamical System (Core Novelty)', bg=colors.HexColor('#005B5B')))
story.append(sp(5))
story.append(Paragraph(
    'Layer 3 gives each hidden state momentum using the Euler Method over a 10-second interval Δt. '
    'Three timescales are modelled simultaneously:', body_sty))
story.append(sp(6))

def dyn_block(title, eq, rows, note=None, bg=CYAN_BG, hdr_bg=CYAN_DARK):
    content = []
    # title bar
    tb = Table([[Paragraph(title, sty('Normal', fontSize=9, textColor=WHITE, fontName='Helvetica-Bold'))]], 
               colWidths=[W-36*mm])
    tb.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), hdr_bg),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
    ]))
    content.append(tb)
    # equation
    et = Table([[Paragraph(eq, sty('Normal', fontSize=10, textColor=NAVY, fontName='Courier-Bold', alignment=TA_CENTER))]], 
               colWidths=[W-36*mm])
    et.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), bg),
        ('TOPPADDING', (0,0), (-1,-1), 7),
        ('BOTTOMPADDING', (0,0), (-1,-1), 7),
    ]))
    content.append(et)
    # param rows
    cw = [(W-36*mm)*f for f in [0.22, 0.12, 0.66]]
    pr = []
    for r in rows:
        pr.append([
            Paragraph(r[0], sty('Normal', fontSize=8, textColor=GREY_DARK, fontName='Courier')),
            Paragraph(r[1], sty('Normal', fontSize=8, textColor=NAVY, fontName='Helvetica-Bold', alignment=TA_CENTER)),
            Paragraph(r[2], sty('Normal', fontSize=8, textColor=BLACK, fontName='Helvetica')),
        ])
    pt = Table(pr, colWidths=cw)
    pt.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), bg),
        ('GRID', (0,0), (-1,-1), 0.3, GREY_MED),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 7),
        ('RIGHTPADDING', (0,0), (-1,-1), 7),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ]))
    content.append(pt)
    if note:
        nt = Table([[Paragraph(note, sty('Normal', fontSize=7.5, textColor=GREY_DARK, fontName='Helvetica-Oblique'))]], 
                   colWidths=[W-36*mm])
        nt.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), GREY_LIGHT),
            ('TOPPADDING', (0,0), (-1,-1), 4),
            ('BOTTOMPADDING', (0,0), (-1,-1), 4),
            ('LEFTPADDING', (0,0), (-1,-1), 8),
        ]))
        content.append(nt)
    return KeepTogether(content)

# 3a Slow Dynamics — Cognitive Capacity C<sub>t</sub> (Years timescale)
story.append(Paragraph('<b>3a. Slow Dynamics — Cognitive Capacity C<sub>t</sub> (Years timescale)</b>', h2_sty))
story.append(dyn_block(
    'dC<sub>t</sub> = \u03b1 ( \u03bc \u2212 C<sub>t</sub> ) dt + \u03c3 dW<sub>t</sub>',
    'dC<sub>t</sub> = \u03b1 ( \u03bc \u2212 C<sub>t</sub> ) dt + \u03c3 dW<sub>t</sub>',
    [
        ['\u03b1 (Recovery Speed)', '0.01', 'very slow; capacity takes hours to recharge'],
        ['\u03bc (Target Baseline)', '1.0', 'normally; drops toward 0 when sleep deficit is detected in X<sub>t</sub>'],
        ['\u03c3 dW<sub>t</sub>', '—', 'Stochastic noise term (Wiener process) — models biological variability'],
        ['C<sub>t</sub> in our code', '1.0', 'Assumed full energy in the worked example; no fatigue sensor yet'],
    ],
    bg=CYAN_BG, hdr_bg=CYAN_DARK
))
story.append(sp(6))

# 3b Fast Dynamics — Cognitive Demand D<sub>t</sub> (Hours timescale)
story.append(Paragraph('<b>3b. Fast Dynamics — Cognitive Demand D<sub>t</sub> (Hours timescale)</b>', h2_sty))
story.append(dyn_block(
    'dD<sub>t</sub> = [ f(work) \u2212 \u03bb ( D<sub>t</sub> \u2212 0.1 ) ] dt + \u03c3 dW<sub>t</sub>',
    'dD<sub>t</sub> = [ f(work) \u2212 \u03bb ( D<sub>t</sub> \u2212 0.1 ) ] dt + \u03c3 dW<sub>t</sub>',
    [
        ['f(work)', '—', 'Work Pressure — computed from Task Switches + Notifications count in X<sub>t</sub>'],
        ['\u03bb (Cooling Rate)', '0.2', 'stress decays over a few minutes of inactivity'],
        ['0.1', '—', 'Resting baseline; demand never drops below idle level'],
        ['Notifications role', '—', 'Each audio peak detected by our sensor increments f(work), raising D<sub>t</sub>'],
    ],
    bg=MAGENTA_BG, hdr_bg=MAGENTA
))
story.append(sp(6))

# 3c Medium Dynamics — Habit State H<sub>t</sub> (Weeks timescale)
story.append(Paragraph('<b>3c. Medium Dynamics — Habit State H<sub>t</sub> (Weeks timescale)</b>', h2_sty))
story.append(dyn_block(
    'H<sub>t+1</sub> = H<sub>t</sub> + \u03b7 \u00b7 R<sub>t</sub>',
    'H<sub>t+1</sub> = H<sub>t</sub> + \u03b7 \u00b7 R<sub>t</sub>',
    [
        ['\u03b7 (Learning Rate)', '0.1', 'habit changes gradually over weeks'],
        ['R<sub>t</sub> (Reward)', '+0.05', 'for good security behaviour (password manager used)'],
        ['R<sub>t</sub> (Penalty)', '\u22120.10', 'for bad security behaviour (manual password typed, HTTP site)'],
        ['H<sub>t</sub> in example', '0.8', 'user has mostly good habits'],
    ],
    bg=GREEN_BG, hdr_bg=GREEN_DARK
))
story.append(sp(6))

# 3d CRG
story.append(Paragraph('<b>3d. Cognitive Reserve Gap — CRG<sub>t</sub> (The Core Logic)</b>', h2_sty))
crg_data = [
    [Paragraph('CRG<sub>t</sub> = C<sub>t</sub> \u2212 D<sub>t</sub>',
               sty('Normal', fontSize=12, textColor=NAVY, fontName='Courier-Bold', alignment=TA_CENTER)),
     Paragraph('', body_sty)],
    [Paragraph('<b>Positive CRG<sub>t</sub></b>', sty('Normal', fontSize=8, textColor=GREEN_DARK, fontName='Helvetica-Bold')),
     Paragraph('Spare brainpower available \u2192 Healthy, lower risk', sty('Normal', fontSize=8, textColor=GREEN_DARK, fontName='Helvetica'))],
    [Paragraph('<b>Negative CRG<sub>t</sub></b>', sty('Normal', fontSize=8, textColor=RED_DARK, fontName='Helvetica-Bold')),
     Paragraph('Cognitive debt \u2192 Risky, higher mistake probability', sty('Normal', fontSize=8, textColor=RED_DARK, fontName='Helvetica'))],
    [Paragraph('<i>Note (PDF sign convention)</i>', small_sty),
     Paragraph('The PDF defines CRG<sub>t</sub> = D<sub>t</sub>\u2212 C<sub>t</sub> in Layer 4 formula; the \u03b2<sub>1</sub> weight is negative (\u22123) to flip direction correctly',
               sty('Normal', fontSize=7.5, textColor=GREY_DARK, fontName='Helvetica-Oblique'))],
]
crg_tbl = Table(crg_data, colWidths=[(W-36*mm)*0.3, (W-36*mm)*0.7])
crg_tbl.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,0), GOLD_BG),
    ('BACKGROUND', (0,1), (-1,1), GREEN_BG),
    ('BACKGROUND', (0,2), (-1,2), RED_BG),
    ('BACKGROUND', (0,3), (-1,3), GREY_LIGHT),
    ('GRID', (0,0), (-1,-1), 0.4, GREY_MED),
    ('SPAN', (0,0), (1,0)),
    ('TOPPADDING', (0,0), (-1,-1), 6),
    ('BOTTOMPADDING', (0,0), (-1,-1), 6),
    ('LEFTPADDING', (0,0), (-1,-1), 8),
    ('RIGHTPADDING', (0,0), (-1,-1), 8),
    ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
]))
story.append(crg_tbl)
story.append(sp(10))

# ══════════════════════════════════════════════════════════════
# LAYER 4
# ══════════════════════════════════════════════════════════════
story.append(section_header('Layer 4 — Cyber Risk Forecasting Engine', bg=colors.HexColor('#6B0000')))
story.append(sp(5))
story.append(Paragraph(
    'Layer 4 converts the hidden state vector Z<sub>t</sub> into a single Mistake Probability P(M<sub>t</sub>) '
    'using a logistic regression over the three key roots.', body_sty))
story.append(sp(6))

story.append(Paragraph('<b>Step 1 — Logit Score (Linear Combination with Beta Weights):</b>', h2_sty))

score_data = [
    [Paragraph('Score = \u03b2<sub>1</sub> \u00b7 CRG<sub>t</sub> + \u03b2<sub>2</sub> \u00b7 H<sub>t</sub> + \u03b2<sub>3</sub> \u00b7 A<sub>t</sub> + bias',
               sty('Normal', fontSize=10, textColor=colors.HexColor('#6B0000'), fontName='Courier-Bold', alignment=TA_CENTER)),
     Paragraph('', body_sty)],
    [Paragraph('Score = ( \u22123 \u00d7 CRG<sub>t</sub> ) + ( \u22123 \u00d7 H<sub>t</sub> ) + ( 6 \u00d7 A<sub>t</sub> ) + 0.5',
               sty('Normal', fontSize=9.5, textColor=NAVY, fontName='Courier-Bold', alignment=TA_CENTER)),
     Paragraph('', body_sty)],
    [Paragraph('\u03b2<sub>1</sub> = \u22123', sty('Normal', fontSize=8, textColor=CYAN_DARK, fontName='Courier-Bold')),
     Paragraph('Reserve Gap weight — higher spare capacity lowers risk (negative sign)',
               sty('Normal', fontSize=8, textColor=BLACK, fontName='Helvetica'))],
    [Paragraph('\u03b2<sub>2</sub> = \u22123', sty('Normal', fontSize=8, textColor=GREEN_DARK, fontName='Courier-Bold')),
     Paragraph('Habit Quality weight — better habits lower risk (negative sign)',
               sty('Normal', fontSize=8, textColor=BLACK, fontName='Helvetica'))],
    [Paragraph('\u03b2<sub>3</sub> = +6', sty('Normal', fontSize=8, textColor=RED_DARK, fontName='Courier-Bold')),
     Paragraph('Adversarial Threat weight — active attack massively raises risk (double weight)',
               sty('Normal', fontSize=8, textColor=BLACK, fontName='Helvetica'))],
    [Paragraph('Bias = +0.5', sty('Normal', fontSize=8, textColor=GREY_DARK, fontName='Courier-Bold')),
     Paragraph('Baseline offset — slight positive risk even in ideal conditions',
               sty('Normal', fontSize=8, textColor=BLACK, fontName='Helvetica'))],
]
score_tbl = Table(score_data, colWidths=[(W-36*mm)*0.3, (W-36*mm)*0.7])
score_tbl.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,0), RED_BG),
    ('BACKGROUND', (0,1), (-1,1), GOLD_BG),
    ('BACKGROUND', (0,2), (-1,2), CYAN_BG),
    ('BACKGROUND', (0,3), (-1,3), GREEN_BG),
    ('BACKGROUND', (0,4), (-1,4), RED_BG),
    ('BACKGROUND', (0,5), (-1,5), GREY_LIGHT),
    ('SPAN', (0,0), (1,0)),
    ('SPAN', (0,1), (1,1)),
    ('GRID', (0,0), (-1,-1), 0.4, GREY_MED),
    ('TOPPADDING', (0,0), (-1,-1), 7),
    ('BOTTOMPADDING', (0,0), (-1,-1), 7),
    ('LEFTPADDING', (0,0), (-1,-1), 10),
    ('RIGHTPADDING', (0,0), (-1,-1), 10),
    ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
]))
story.append(score_tbl)
story.append(sp(6))

story.append(Paragraph('<b>Step 2 — Sigmoid Squash to Probability:</b>', h2_sty))
sig_tbl = Table([[Paragraph(
    'P(M<sub>t</sub>) = \u03c3( Score ) = 1 / ( 1 + e<super>\u2212Score</super> )',
    sty('Normal', fontSize=12, textColor=colors.HexColor('#6B0000'), fontName='Courier-Bold', alignment=TA_CENTER)
)]], colWidths=[W-36*mm])
sig_tbl.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,-1), RED_BG),
    ('BOX', (0,0), (-1,-1), 1, RED_DARK),
    ('TOPPADDING', (0,0), (-1,-1), 10),
    ('BOTTOMPADDING', (0,0), (-1,-1), 10),
]))
story.append(sig_tbl)
story.append(sp(10))

# ══════════════════════════════════════════════════════════════
# WORKED EXAMPLE
# ══════════════════════════════════════════════════════════════
story.append(section_header('Worked Example — High-Load Simulation Scenario', bg=colors.HexColor('#1A3A5C')))
story.append(sp(5))
story.append(Paragraph(
    'We solve a high-stress cognitive scenario using the LCDT symbols and formulas. '
    'This example demonstrates how multi-modal pressure translates into a concrete risk score.', body_sty))
story.append(sp(8))

def case_block(title, bg_hdr, bg_body, items):
    rows = [[Paragraph(title, sty('Normal', fontSize=10, textColor=WHITE, fontName='Helvetica-Bold'))]]
    tbl = Table(rows, colWidths=[W-36*mm])
    tbl.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), bg_hdr),
        ('TOPPADDING', (0,0), (-1,-1), 6), ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
    ]))
    body_rows = []
    for label, val, note in items:
        body_rows.append([
            Paragraph(label, sty('Normal', fontSize=8, textColor=GREY_DARK, fontName='Courier')),
            Paragraph(val, sty('Normal', fontSize=8, textColor=NAVY, fontName='Helvetica-Bold')),
            Paragraph(note, sty('Normal', fontSize=8, textColor=BLACK, fontName='Helvetica')),
        ])
    body_tbl = Table(body_rows, colWidths=[(W-36*mm)*f for f in [0.3, 0.2, 0.5]])
    body_tbl.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), bg_body),
        ('GRID', (0,0), (-1,-1), 0.3, GREY_MED),
        ('TOPPADDING', (0,0), (-1,-1), 5), ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 8), ('RIGHTPADDING', (0,0), (-1,-1), 8),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ]))
    return KeepTogether([tbl, body_tbl])

story.append(case_block(
    'High-Stress Scenario (Active Task Switching & Phishing Interaction)',
    GREEN_DARK, GREEN_BG,
    [
        ('Layer 1 — X<sub>t</sub>[Task Switches]', '20', 'user jumping between windows'),
        ('Layer 1 — X<sub>t</sub>[Notifications]', '10', 'captured from audio sensors'),
        ('Layer 1 — X<sub>t</sub>[Phishing]', '1', 'user clicked unknown email link'),
        ('f(work)', '0.5', 'normalized from 20 task switches + 10 notification peaks'),
        ('Initial States', '1.0, 0.1, 0.8, 0.0', 'Capacity, Demand, Habits, Adversarial'),
        ('New Demand (D<sub>t+1</sub>)', '0.6', '0.1 + [0.5 \u2212 0.2(0.1\u22120.1)] = 0.6'),
        ('Adversarial (A<sub>t+1</sub>)', '1.0', 'phishing click instant spike'),
        ('Habits (H<sub>t+1</sub>)', '0.79', '0.8 + 0.1 \u00d7 (\u22120.1) — slight penalty for phishing click'),
        ('Reserve Gap (CRG<sub>t</sub>)', '0.4', '1.0 \u2212 0.6 = 0.4'),
        ('Score', '2.9', '(\u22123\u00d70.4)+(\u22123\u00d70.8)+(6\u00d71.0)+0.5 = \u22121.2\u22122.4+6.0+0.5'),
    ]
))

# Final result callout
story.append(sp(6))
final_tbl = Table([[Paragraph(
    'FINAL RESULT: P(M<sub>t</sub>) = 94.7% Mistake Risk',
    sty('Normal', fontSize=13, textColor=WHITE, fontName='Helvetica-Bold', alignment=TA_CENTER)
)]], colWidths=[W-36*mm])
final_tbl.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,-1), RED_DARK),
    ('BOX', (0,0), (-1,-1), 2, colors.HexColor('#FF4444')),
    ('TOPPADDING', (0,0), (-1,-1), 12),
    ('BOTTOMPADDING', (0,0), (-1,-1), 12),
]))
story.append(final_tbl)
story.append(sp(10))

# ══════════════════════════════════════════════════════════════
# LAYER 5
# ══════════════════════════════════════════════════════════════
story.append(section_header('Layer 5 — Digital Twin Simulation Engine: E[ M<sub>t:T</sub> | do(\u03c0) ]', bg=colors.HexColor('#2E4A1E')))
story.append(sp(5))
story.append(Paragraph(
    'Layer 5 is the counterfactual brain of the Digital Twin. The notation <b>do(\u03c0)</b> represents a forced '
    'Policy Intervention simulated over 30 days to answer \'What If?\' questions.',
    body_sty))
story.append(sp(6))

l5_hdr = [
    Paragraph('<b>Intervention do(\u03c0)</b>', sty('Normal', fontSize=8, textColor=WHITE, fontName='Helvetica-Bold')),
    Paragraph('<b>What We Force</b>', sty('Normal', fontSize=8, textColor=WHITE, fontName='Helvetica-Bold')),
    Paragraph('<b>Effect on Z<sub>t</sub></b>', sty('Normal', fontSize=8, textColor=WHITE, fontName='Helvetica-Bold')),
    Paragraph('<b>Expected P(M<sub>t</sub>)</b>', sty('Normal', fontSize=8, textColor=WHITE, fontName='Helvetica-Bold', alignment=TA_CENTER)),
]
def l5r(a, b, c, d, dcolor=BLACK):
    return [
        Paragraph(a, sty('Normal', fontSize=8, textColor=BLACK, fontName='Helvetica-Bold')),
        Paragraph(b, sty('Normal', fontSize=8, textColor=BLACK, fontName='Helvetica')),
        Paragraph(c, sty('Normal', fontSize=8, textColor=BLACK, fontName='Helvetica')),
        Paragraph(d, sty('Normal', fontSize=8, textColor=dcolor, fontName='Helvetica-Bold', alignment=TA_CENTER)),
    ]

l5_rows = [
    l5_hdr,
    l5r('do(UX Optimisation)', 'Cap X<sub>t</sub>[Notifications]=0, Task Switches=0',
        'D<sub>t</sub> stays near 0.1 (resting); CRG<sub>t</sub>\u21920.9 (very healthy)', '\u2193 drops significantly', GREEN_DARK),
    l5r('do(Security Training)', 'Force daily \u03b7\u00b7R<sub>t</sub> spike in H<sub>t</sub> formula',
        'H<sub>t</sub>\u21921.0 over ~30 days; \u03b2<sub>2</sub>\u00b7H<sub>t</sub> term maximally protective', '\u2193 moderate drop', GREEN_DARK),
    l5r('do(Increased Workload)', 'Force f(work)=1.0 (max) for all 30 days',
        'D<sub>t</sub> rises; CRG<sub>t</sub> goes negative \u2192 Cognitive Debt', '\u2191 risk escalates', RED_DARK),
    l5r('do(Aging Progression)', 'Decrease \u03bc (Capacity Baseline) by 2% per simulated day',
        'C<sub>t</sub> ceiling drops slowly; CRG<sub>t</sub> shrinks even at same workload', '\u2191 gradual rise', colors.HexColor('#884400')),
    l5r('Baseline (no intervention)', 'Current behaviour continues unchanged',
        'If already at cognitive debt, line climbs as C<sub>t</sub> drains', '\u2192 natural trajectory', GREY_DARK),
]
cw5 = [(W-36*mm)*f for f in [0.22, 0.25, 0.35, 0.18]]
l5_tbl = Table(l5_rows, colWidths=cw5)
l5_tbl.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2E4A1E')),
    ('BACKGROUND', (0,1), (-1,1), GREEN_BG),
    ('BACKGROUND', (0,2), (-1,2), colors.HexColor('#F0FFF0')),
    ('BACKGROUND', (0,3), (-1,3), RED_BG),
    ('BACKGROUND', (0,4), (-1,4), colors.HexColor('#FFF5E6')),
    ('BACKGROUND', (0,5), (-1,5), GREY_LIGHT),
    ('GRID', (0,0), (-1,-1), 0.4, GREY_MED),
    ('TOPPADDING', (0,0), (-1,-1), 6),
    ('BOTTOMPADDING', (0,0), (-1,-1), 6),
    ('LEFTPADDING', (0,0), (-1,-1), 7),
    ('RIGHTPADDING', (0,0), (-1,-1), 7),
    ('VALIGN', (0,0), (-1,-1), 'TOP'),
]))
story.append(l5_tbl)
story.append(sp(10))

# ══════════════════════════════════════════════════════════════
# GLOSSARY
# ══════════════════════════════════════════════════════════════
story.append(section_header('Complete Symbol Glossary', bg=colors.HexColor('#2A2A4A')))
story.append(sp(5))

gloss = [
    ('X<sub>t</sub>', 'Observation vector at time t', 'All 12 raw sensor readings every 10 seconds, including notification audio peaks'),
    ('Z<sub>t</sub>', 'Latent hidden state', 'The four internal cognitive-cyber roots inferred from X<sub>t</sub>'),
    ('C<sub>t</sub>', 'Cognitive Capacity', 'Mental energy tank (cyan). Starts at 1.0, drains slowly with fatigue/age'),
    ('D<sub>t</sub>', 'Cognitive Demand', 'Stress load (magenta). Raised by notifications and task-switching'),
    ('H<sub>t</sub>', 'Habit State', 'Security behaviour quality (green). Updated by \u03b7\u00b7R<sub>t</sub> each interval'),
    ('A<sub>t</sub>', 'Adversarial Exposure', 'Threat level (red). Spikes instantly to 1.0 on phishing click'),
    ('CRG<sub>t</sub>', 'Cognitive Reserve Gap', 'C<sub>t</sub>\u2212 D<sub>t</sub>. Positive = healthy buffer. Negative = cognitive debt'),
    ('P(M<sub>t</sub>)', 'Mistake Probability', 'Final risk output, 0\u2013100%. \u03c3(\u03b2<sub>1</sub>\u00b7CRG + \u03b2<sub>2</sub>\u00b7H + \u03b2<sub>3</sub>\u00b7A)'),
    ('do(\u03c0)', 'Policy Intervention', 'A forced change applied in Layer 5 counterfactual simulation'),
    ('\u03b1', 'Recovery Speed', '0.01 — rate at which C<sub>t</sub> returns to baseline \u03bc'),
    ('\u03bc', 'Target Capacity Baseline', '1.0 normally; reduced when sleep deficit detected in X<sub>t</sub>'),
    ('\u03bb', 'Cooling Rate', '0.2 — rate at which D<sub>t</sub> decays back to resting level 0.1'),
    ('\u03b7', 'Learning Rate', '0.1 — how quickly H<sub>t</sub> responds to positive/negative behaviour'),
    ('R<sub>t</sub>', 'Behaviour Reward', '+0.05 for good security; \u22120.10 for bad security actions'),
    ('\u03b2<sub>1,2,3</sub>', 'Beta Weights', 'Importance factors: \u03b2<sub>1</sub>=\u22123 (CRG), \u03b2<sub>2</sub>=\u22123 (H), \u03b2<sub>3</sub>=+6 (A)'),
    ('\u03c3(\u00b7)', 'Sigmoid function', '1/(1+e<sup>-x</sup>); squashes any score into 0\u20131 probability'),
    ('dW<sub>t</sub>', 'Wiener Process', 'Stochastic noise term modelling biological/behavioural randomness'),
]

gloss_hdr = [
    Paragraph('<b>Symbol</b>', sty('Normal', fontSize=8, textColor=WHITE, fontName='Helvetica-Bold')),
    Paragraph('<b>Full Name</b>', sty('Normal', fontSize=8, textColor=WHITE, fontName='Helvetica-Bold')),
    Paragraph('<b>Meaning in This Workflow</b>', sty('Normal', fontSize=8, textColor=WHITE, fontName='Helvetica-Bold')),
]
gloss_rows = [gloss_hdr]
for sym, name, meaning in gloss:
    gloss_rows.append([
        Paragraph(sym, sty('Normal', fontSize=8, textColor=NAVY, fontName='Courier-Bold')),
        Paragraph(name, sty('Normal', fontSize=8, textColor=BLACK, fontName='Helvetica-Bold')),
        Paragraph(meaning, sty('Normal', fontSize=8, textColor=BLACK, fontName='Helvetica')),
    ])
cw_g = [(W-36*mm)*f for f in [0.12, 0.22, 0.66]]
gloss_tbl = Table(gloss_rows, colWidths=cw_g)
gloss_tbl.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2A2A4A')),
    ('ROWBACKGROUNDS', (0,1), (-1,-1), [GREY_LIGHT, WHITE]),
    ('GRID', (0,0), (-1,-1), 0.3, GREY_MED),
    ('TOPPADDING', (0,0), (-1,-1), 5),
    ('BOTTOMPADDING', (0,0), (-1,-1), 5),
    ('LEFTPADDING', (0,0), (-1,-1), 7),
    ('RIGHTPADDING', (0,0), (-1,-1), 7),
    ('VALIGN', (0,0), (-1,-1), 'TOP'),
]))
story.append(gloss_tbl)
story.append(sp(10))

# ══════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════
story.append(section_header('Summary: The Digital Twin Mapping', bg=GREEN_DARK))
story.append(sp(5))
sum_tbl = Table([[Paragraph(
    'The Cyber Watchdog translates raw sensor signals into a predictive Digital Twin using 5 mathematical layers. '
    'By capturing activity like task-switching and notifications, the system builds a live model of '
    'Cognitive Demand (D<sub>t</sub>) and Capacity (C<sub>t</sub>). This allows the system to accurately forecast '
    'the Mistake Probability P(M<sub>t</sub>) and simulate long-term security outcomes using the Layer 5 engine, '
    'ensuring a proactive and privacy-first approach to cybersecurity.',
    sty('Normal', fontSize=9, leading=14, textColor=colors.HexColor('#0A2A0A'), fontName='Helvetica')
)]], colWidths=[W-36*mm])
sum_tbl.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,-1), GREEN_BG),
    ('BOX', (0,0), (-1,-1), 1.5, GREEN_DARK),
    ('TOPPADDING', (0,0), (-1,-1), 12),
    ('BOTTOMPADDING', (0,0), (-1,-1), 12),
    ('LEFTPADDING', (0,0), (-1,-1), 14),
    ('RIGHTPADDING', (0,0), (-1,-1), 14),
]))
story.append(sum_tbl)
story.append(sp(8))

# ══════════════════════════════════════════════════════════════
# COMPREHENSIVE 5-LAYER WALKTHROUGH EXAMPLE
# ══════════════════════════════════════════════════════════════
story.append(section_header('Full 5-Layer Algorithm Walkthrough: "Late-Night Crunch"', bg=NAVY))
story.append(sp(8))

def example_step(layer_num, title, items, bg=GREY_LIGHT, border=GREY_MED):
    content = []
    # Layer Title
    content.append(Table([[Paragraph(f'<b>Layer {layer_num} — {title}</b>', 
                   sty('Normal', fontSize=10, textColor=NAVY, fontName='Helvetica-Bold'))]], 
                   colWidths=[W-36*mm], style=[('BACKGROUND', (0,0), (-1,-1), bg), 
                                                ('TOPPADDING', (0,0), (-1,-1), 6), ('BOTTOMPADDING', (0,0), (-1,-1), 6),
                                                ('LEFTPADDING', (0,0), (-1,-1), 10)]))
    
    body_rows = []
    for step, desc, why in items:
        body_rows.append([
            Paragraph(f'<b>{step}</b>', sty('Normal', fontSize=8.5, textColor=BLACK, fontName='Helvetica-Bold')),
            Paragraph(desc, sty('Normal', fontSize=8, textColor=NAVY, fontName='Courier-Bold')),
            Paragraph(why, sty('Normal', fontSize=8, textColor=BLACK, fontName='Helvetica-Oblique')),
        ])
    
    tbl = Table(body_rows, colWidths=[(W-36*mm)*f for f in [0.25, 0.20, 0.55]])
    tbl.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), WHITE),
        ('GRID', (0,0), (-1,-1), 0.5, border),
        ('TOPPADDING', (0,0), (-1,-1), 6), ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 8), ('RIGHTPADDING', (0,0), (-1,-1), 8),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ]))
    content.append(tbl)
    content.append(sp(6))
    return KeepTogether(content)

# Layer 1 Example
story.append(example_step(1, 'Observation (X<sub>t</sub>)', [
    ('Input FAMILY', 'Signals Captured', 'Reasoning (Why)'),
    ('Work Context', 'Task Switches = 25', 'High volume context switching increases Demand floor.'),
    ('Distractions', 'Notifications = 12', 'Audio pings directly increment the work pressure function.'),
    ('Biomarkers', 'Time = 23:45 (Night)', 'Biological clock signals a drop in Energy baseline (\u03bc).'),
    ('Threat Familien', 'Phishing = 1.0 (Click)', 'Adversarial family triggers max threat spike.'),
], bg=CYAN_BG, border=CYAN_DARK))

# Layer 2 Example
story.append(example_step(2, 'State Inference (Z<sub>t</sub>)', [
    ('Calculation', 'Result', 'Logic (What)'),
    ('Work Pressure', 'f(work) = 0.65', 'Normalized switches (25/40) + notifications (12/10) capped.'),
    ('Energy Floor', '\u03bc = 1.0 \u2212 0.2 = 0.8', 'Night-shift penalty applied to standard 1.0 baseline.'),
    ('Habit Reward', 'R<sub>t</sub> = \u22120.1', 'Penalty assigned for clicking a suspicious link.'),
    ('Inference P(Z)', 'Z = [C, D, H, A]', 'Mapping raw X<sub>t</sub> into the 4 latent hidden roots.'),
], bg=PURPLE_BG, border=PURPLE))

# Layer 3 Example
story.append(example_step(3, 'Dynamics (Euler Step Equations)', [
    ('Formula / Step', 'Arithmetic Solving', 'Dynamic Effect'),
    ('dD<sub>t</sub> Eq.', '(0.65 \u2212 0.2 \u00d7 (0.1\u22120.1)) \u00d7 1.0', 'Demand increases from 0.1 to 0.75 instantly.'),
    ('dC<sub>t</sub> Eq.', '0.01 \u00d7 (0.8 \u2212 1.0) \u00d7 1.0', 'Capacity drains from 1.0 to 0.98 due to fatigue.'),
    ('H<sub>t+1</sub> Eq.', '0.8 + 0.1 \u00d7 (\u22120.1)', 'Habits drop from 0.8 to 0.79 (Protective Shield weak).'),
    ('CRG<sub>t</sub> Gap', 'C<sub>t</sub> \u2212 D<sub>t</sub> = 0.98 \u2212 0.75', 'Positive buffer is only 0.23 (Critical level).'),
], bg=GOLD_BG, border=GOLD))

# Layer 4 Example
story.append(example_step(4, 'Risk Score (Logistic Regression)', [
    ('Weight Mapping', 'Step-by-Step Summation', 'Statistical Meaning'),
    ('\u03b2\u2081 \u00b7 CRG', '\u22123 \u00d7 0.23 = \u22120.69', 'Reserve Gap impact (Low reserve = More risk).'),
    ('\u03b2\u2082 \u00b7 H', '\u22123 \u00d7 0.79 = \u22122.37', 'Habit impact (Weak habits = More risk).'),
    ('\u03b2\u2083 \u00b7 A', '6 \u00d7 1.0 = 6.0', 'Threat impact (Active attack = Max risk).'),
    ('Final Logit', '\u22120.69 \u2212 2.37 + 6.0 + 0.5', 'Logit Score = 3.44 (Severe danger zone).'),
    ('P(M<sub>t</sub>)', '1 / (1 + e<sup>\u22123.44</sup>)', 'Result: 96.88% probability of a critical mistake.'),
], bg=RED_BG, border=RED_DARK))

# Layer 5 Example
story.append(example_step(5, 'Simulation (Counterfactual do-calculus)', [
    ('What-If Scenario', 'Formula Change', 'Predicted Impact'),
    ('do(UX Optim)', 'Force X<sub>t</sub>[Alerts] = 0', 'Eliminating interruptions to preserve the gap.'),
    ('New dD<sub>t</sub>', '(0.25 \u2212 0.2 \u00d7 0) \u00d7 1.0', 'New demand would be 0.35 instead of 0.75.'),
    ('New Score', '\u22123(0.63) \u2212 3(0.79) + 6(1) + 0.5', 'New Score = 2.24 (Medium danger zone).'),
    ('Simulation Result', 'P(M<sub>t</sub>) \u2192 90.3%', 'Even under attack, UX fix saves ~6% risk immediately.'),
], bg=GREEN_BG, border=GREEN_DARK))

story.append(sp(4))

footer_tbl = Table([[Paragraph(
    'Lifelong Cognitive–Cyber Digital Twin (LCDT) · Cyber Watchdog Framework',
    sty('Normal', fontSize=8, textColor=GREY_DARK, fontName='Helvetica-Oblique', alignment=TA_CENTER)
)]], colWidths=[W-36*mm])
footer_tbl.setStyle(TableStyle([
    ('TOPBORDER', (0,0), (-1,-1), 0.5, GREY_MED),
    ('TOPPADDING', (0,0), (-1,-1), 6),
]))
story.append(footer_tbl)

doc.build(story)
print("Done")