"""
Hybrid Model Flowchart - Validator + Corporate Finance Structure (Graphviz)
"""
import graphviz
from pathlib import Path

# Create directed graph
dot = graphviz.Digraph('Hybrid Model', format='pdf')
dot.attr(rankdir='TB', size='10,6', dpi='300')
dot.attr('node', shape='box', style='rounded,filled', fontsize='12', fontname='Helvetica')

# Define colors
colors = {
    'equity': '#9945FF',      # SOL Purple
    'debt': '#58A6FF',        # Accent Blue
    'validator': '#14F195',   # SOL Green
    'revenue': '#3FB950',     # Success Green
    'output': '#F0883E',      # Warning Orange
}

# Capital Sources (Top)
with dot.subgraph(name='cluster_capital') as c:
    c.attr(label='Capital Sources', style='dashed', color='gray')
    c.node('equity', 'Equity Capital\n$50M', fillcolor=colors['equity'], fontcolor='white')
    c.node('convert', 'Convertible Bonds\n$100M (0% coupon)', fillcolor=colors['debt'], fontcolor='white')
    c.node('pipe', 'PIPE Proceeds\n$30M', fillcolor=colors['debt'], fontcolor='white')

# Treasury (Middle)
dot.node('treasury', 'SOL Treasury\n1.2M SOL', shape='cylinder', fillcolor='#161B22', fontcolor='white')

# Validator Operations
with dot.subgraph(name='cluster_validator') as c:
    c.attr(label='Validator Operations', style='dashed', color='gray')
    c.node('validator', 'Validator Node\n500K Self-Stake', fillcolor=colors['validator'], fontcolor='black')
    c.node('lst', 'LST Issuance\n(optional)', fillcolor=colors['validator'], fontcolor='black')

# Revenue Streams
with dot.subgraph(name='cluster_revenue') as c:
    c.attr(label='Revenue Streams', style='dashed', color='gray')
    c.node('staking', 'Staking Rewards\n7.5% APY', fillcolor=colors['revenue'], fontcolor='black')
    c.node('commission', 'Commission\n5% on delegated', fillcolor=colors['revenue'], fontcolor='black')
    c.node('mev', 'MEV Tips\n(Jito)', fillcolor=colors['revenue'], fontcolor='black')

# Yield Enhancement
dot.node('options', 'Options Overlay\nCovered Calls', fillcolor=colors['output'], fontcolor='black')

# Output
dot.node('output', 'Target:\n15-25% Annual ROE', shape='doubleoctagon', fillcolor='#9945FF', fontcolor='white')

# Edges - Capital Flow
dot.edge('equity', 'treasury', label='Direct\nInvestment')
dot.edge('convert', 'treasury', label='Bond\nProceeds')
dot.edge('pipe', 'treasury', label='Private\nPlacement')

# Edges - Treasury to Operations
dot.edge('treasury', 'validator', label='Self-Stake')
dot.edge('treasury', 'lst', label='Optional\nLST Issuance', style='dashed')

# Edges - Revenue Generation
dot.edge('validator', 'staking')
dot.edge('validator', 'commission')
dot.edge('validator', 'mev')

# Edges - Options Overlay
dot.edge('treasury', 'options', label='Collateral')
dot.edge('options', 'output', label='Premium\nIncome')

# Edges - Revenue to Output
dot.edge('staking', 'output')
dot.edge('commission', 'output')
dot.edge('mev', 'output')

# Render
output_path = Path(__file__).parent / 'chart'
dot.render(str(output_path), cleanup=True)

print("Chart saved: 11_hybrid_model_flow/chart.pdf")
