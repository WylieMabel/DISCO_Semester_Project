# Research Plan: LLM Simulation of Deliberation using America in One Room

**Dataset**: America in One Room (Fishkin et al., 2021, APSR)  
**Paradigm**: Deliberative Polling (NOT traditional Delphi)  
**Core Strength**: Rich demographics + large sample  
**Date**: October 23, 2025

---

## What This Dataset Enables

### ✅ What You CAN Study

1. **Opinion Dynamics After Group Discussion**
   - How opinions shift when people discuss face-to-face
   - Which demographic groups shift more vs less
   - Group composition effects (diverse vs homogeneous)

2. **Behavioral Fingerprints**
   - Democrat vs Republican shift patterns
   - Age, race, gender effects on opinion change
   - "Shifter types" - cluster people by their shift profiles

3. **LLM as Social Simulator**
   - Can LLMs predict individual opinion shifts?
   - Can LLM agents roleplay participants in discussion?
   - Do LLM-generated arguments reproduce human persuasion patterns?

4. **Partisan Depolarization**
   - Do Republicans and Democrats converge after discussion?
   - Which topics show strongest depolarization?
   - What mechanisms drive convergence?

### ❌ What You CANNOT Study

1. **Traditional Delphi Mechanisms**
   - Statistical feedback loops (seeing median/IQR)
   - Anonymous asynchronous deliberation
   - Iterative convergence over multiple rounds

2. **Pure Consensus Formation**
   - No group decision recorded (only individual opinions)
   - No attempt to reach agreement

---

## Revised Research Questions

### Primary Research Questions

**RQ1: Predictive Modeling**
> Can LLMs predict individual opinion shifts in face-to-face deliberation given demographics and initial opinions?

**RQ2: Demographic Correlates**
> Which demographic factors (party, age, race, gender) most strongly predict opinion shift magnitude and direction?

**RQ3: Social Influence**
> Does group composition (partisan diversity, age diversity) affect individual opinion shifts?

**RQ4: LLM Simulation Validity**
> Can LLM agents roleplay participants and reproduce empirical deliberation patterns (depolarization, shift distributions)?

### Secondary Research Questions

**RQ5: Argument Persuasiveness**
> What types of arguments are most persuasive across partisan lines in LLM simulations?

**RQ6: Behavioral Fingerprints**
> Can we identify distinct "deliberator types" based on shift patterns, and do they correlate with demographics?

---

## 4-Phase Implementation Plan

### Phase 1: Data Exploration & Replication (Week 1-2)

#### Objectives
- Understand dataset structure
- Replicate paper's key findings
- Profile opinion shift patterns

#### Tasks

**1.1 Load and Validate Data**
```python
import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv('data/America_in_one_room/copy of Stanford_A1R_Dataset_APSR.csv')

# Extract treatment group (deliberated)
treatment = df[df['POST'] == 1].copy()  # n=1,367

# Get pre/post columns
pre_cols = [c for c in df.columns if c.startswith('Q') and not c.startswith('T2')]  # 50 questions
post_cols = [c for c in df.columns if c.startswith('T2Q')]  # 50 questions

# Demographics
demo_cols = ['GENDER', 'AGE', 'RACETHNICITY', 'PARTYID3', 'EDUCATION']
```

**1.2 Compute Shift Statistics**
```python
# For each question, compute shifts
shifts = treatment[post_cols].values - treatment[pre_cols].values

# Summary statistics
treatment['shift_magnitude'] = np.abs(shifts).sum(axis=1)  # Total shift across all 50 questions
treatment['shift_net'] = shifts.sum(axis=1)  # Net direction (positive = more supportive overall)
treatment['n_shifts'] = (shifts != 0).sum(axis=1)  # How many questions they changed

print(f"Mean shift magnitude: {treatment['shift_magnitude'].mean():.2f}")
print(f"% who shifted on any question: {(treatment['n_shifts'] > 0).mean()*100:.1f}%")
```

**1.3 Replicate Depolarization Finding**
```python
# Group by party
republicans = treatment[treatment['PARTYID3'] == 2]
democrats = treatment[treatment['PARTYID3'] == 1]

# Example: Immigration question (Q2A - Reduce refugee numbers)
rep_pre = republicans['Q2A'].mean()
rep_post = republicans['T2Q2A'].mean()
dem_pre = democrats['Q2A'].mean()
dem_post = democrats['T2Q2A'].mean()

print(f"Republicans: {rep_pre:.2f} → {rep_post:.2f} (shift: {rep_post - rep_pre:.2f})")
print(f"Democrats:   {dem_pre:.2f} → {dem_post:.2f} (shift: {dem_post - dem_pre:.2f})")
print(f"Pre-gap:  {abs(rep_pre - dem_pre):.2f}")
print(f"Post-gap: {abs(rep_post - dem_post):.2f}")
print(f"Depolarization: {abs(rep_pre - dem_pre) > abs(rep_post - dem_post)}")
```

**Deliverables**:
- `analysis/01_eda.ipynb` - Complete exploratory analysis
- `results/shift_summary.csv` - Statistics for all 50 questions
- `results/party_depolarization.csv` - Pre/post gaps by party
- `figures/shift_distributions.png` - Histograms of shift magnitudes

---

### Phase 2: Supervised Prediction (Week 2-3)

#### Objective
Predict post-deliberation opinions from demographics + initial opinions

#### Model Setup

**Features (X)**:
- Demographics: Party ID, age, race, gender, education (one-hot encoded)
- Initial opinions: All 50 pre-deliberation ratings (Q1...Q50)
- Behavioral: Initial extremism (count of 0/10 ratings), initial variance

**Target (Y)**:
- Post-deliberation opinions: All 50 ratings (T2Q1...T2Q50)
- Alternative: Shifts (T2Q* - Q*)

**Models to Compare**:
1. Linear regression (baseline, interpretable)
2. Random forest (captures non-linearities)
3. XGBoost (typically best performance)
4. LLM-enhanced (add persona embeddings)

#### Key Analysis

**Feature Importance**
```python
import shap
from xgboost import XGBRegressor

# Train model
model = XGBRegressor(n_estimators=100, max_depth=6)
model.fit(X_train, y_train)

# SHAP analysis
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Top predictors
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': np.abs(shap_values).mean(axis=0)
}).sort_values('importance', ascending=False)

print("Top 10 predictors:")
print(feature_importance.head(10))
```

**Demographic Effects**
```python
# For each question, regress shift on demographics only
for q_pre, q_post in zip(pre_cols, post_cols):
    shift = treatment[q_post] - treatment[q_pre]
    
    # Regression: shift ~ party + age + race + gender
    X_demo = pd.get_dummies(treatment[demo_cols], drop_first=True)
    
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(X_demo, shift)
    
    # Extract coefficients
    coefs = pd.Series(model.coef_, index=X_demo.columns)
    print(f"\n{q_pre}: Top demographic effects")
    print(coefs.abs().nlargest(3))
```

**Success Metrics**:
- R² ≥ 0.4 (explains 40% of variance)
- RMSE < 2.0 (on 0-10 scale)
- Better than naive baseline (predict no change)

**Deliverables**:
- `models/xgboost_predictor.pkl` - Trained model
- `results/prediction_performance.csv` - R², RMSE by question
- `results/demographic_effects.csv` - Coefficients for all demos × questions
- `figures/shap_summary.png` - Feature importance plot

---

### Phase 3: LLM Agent Simulation (Week 3-5)

#### Objective
Simulate discussion with LLM agents and validate against real shifts

#### Agent Architecture

```python
class DeliberativeAgent:
    """LLM agent that roleplays a participant"""
    
    def __init__(self, demographics, initial_opinions, llm='gpt-4o-mini'):
        self.age = demographics['age']
        self.gender = demographics['gender']
        self.race = demographics['race']
        self.party = demographics['party']
        self.opinions = initial_opinions.copy()  # Dict: {Q1: 7, Q2A: 3, ...}
        self.llm = llm
    
    def generate_argument(self, question_text, position='current'):
        """Generate argument based on current opinion"""
        
        current_rating = self.opinions.get(question_text, 5)
        stance = "support" if current_rating >= 5 else "oppose"
        
        prompt = f"""You are a {self.age}-year-old {self.gender} {self.race} {self.party} voter.

Question: "{question_text}"
Your current opinion: {current_rating}/10 (where 0=strongly oppose, 10=strongly support)

Generate a brief argument (2-3 sentences) explaining why you {stance} this policy.

Argument:"""
        
        response = call_llm(self.llm, prompt, temperature=0.7, max_tokens=100)
        return response
    
    def hear_and_update(self, question_text, arguments_from_others):
        """Update opinion after hearing others' arguments"""
        
        current = self.opinions[question_text]
        
        # Format arguments
        args_text = "\n".join([f"- {arg}" for arg in arguments_from_others])
        
        prompt = f"""You are a {self.age}-year-old {self.gender} {self.race} {self.party} voter in a discussion group.

Question: "{question_text}"
Your current opinion: {current}/10

You've heard these arguments from your group:
{args_text}

After hearing these perspectives, what is your revised opinion (0-10)? 
Think about:
- Arguments that resonated with you
- New information you learned
- Your core values and how they relate

Respond with ONLY a number 0-10, then one sentence explaining your reasoning.

Revised opinion:"""
        
        response = call_llm(self.llm, prompt, temperature=0.3, max_tokens=50)
        
        # Parse response
        try:
            new_opinion = int(re.search(r'\d+', response).group())
            new_opinion = max(0, min(10, new_opinion))
        except:
            new_opinion = current  # Keep unchanged if parse fails
        
        self.opinions[question_text] = new_opinion
        return new_opinion
```

#### Simulation Protocol

```python
# 1. Initialize agents matching real participants
agents = []
for idx, row in treatment.iterrows():
    agent = DeliberativeAgent(
        demographics={
            'age': row['AGE'],
            'gender': 'Male' if row['GENDER']==1 else 'Female',
            'race': race_map[row['RACETHNICITY']],
            'party': party_map[row['PARTYID3']]
        },
        initial_opinions={q: row[q] for q in pre_cols}
    )
    agents.append(agent)

# 2. Organize by groups (41 small groups)
groups = treatment.groupby('GROUP')

# 3. Simulate discussion for each question
sim_results = []

for question in pre_cols[:10]:  # Start with 10 questions (test)
    question_text = get_question_text(question)  # From codebook
    
    for group_id, group_data in groups:
        group_agents = [agents[i] for i in group_data.index]
        
        # Round 1: Each agent shares initial argument
        arguments = []
        for agent in group_agents:
            arg = agent.generate_argument(question_text)
            arguments.append(arg)
        
        # Round 2: Each agent hears sample of arguments and updates
        for agent in group_agents:
            # Agent hears 3-5 random arguments from group
            heard = random.sample(arguments, min(5, len(arguments)))
            new_opinion = agent.hear_and_update(question_text, heard)
        
        # Record results
        for i, agent_idx in enumerate(group_data.index):
            sim_results.append({
                'agent_id': agent_idx,
                'question': question,
                'real_pre': group_data.iloc[i][question],
                'real_post': group_data.iloc[i][question.replace('Q', 'T2Q')],
                'sim_post': agents[agent_idx].opinions[question_text]
            })

sim_df = pd.DataFrame(sim_results)
```

#### Validation

```python
# Correlation between real and simulated shifts
sim_df['real_shift'] = sim_df['real_post'] - sim_df['real_pre']
sim_df['sim_shift'] = sim_df['sim_post'] - sim_df['real_pre']

from scipy.stats import pearsonr
r, p = pearsonr(sim_df['real_shift'], sim_df['sim_shift'])
print(f"Real vs Sim shift correlation: r={r:.3f}, p={p:.4f}")

# RMSE
rmse = np.sqrt(((sim_df['real_post'] - sim_df['sim_post'])**2).mean())
print(f"RMSE: {rmse:.3f}")

# Depolarization test
rep_sim = sim_df[treatment.loc[sim_df['agent_id'], 'PARTYID3'] == 2]
dem_sim = sim_df[treatment.loc[sim_df['agent_id'], 'PARTYID3'] == 1]

print(f"\nReal depolarization: {(rep_sim['real_shift'].mean(), dem_sim['real_shift'].mean())}")
print(f"Sim depolarization:  {(rep_sim['sim_shift'].mean(), dem_sim['sim_shift'].mean())}")
```

**Success Metrics**:
- Shift correlation r ≥ 0.3
- RMSE < 3.0
- Depolarization direction matches real data
- Similar shift magnitude distributions

**Deliverables**:
- `models/deliberative_agent.py` - Agent class
- `results/simulation_validation.csv` - Real vs sim comparison
- `results/generated_arguments.csv` - Sample arguments by party
- `figures/real_vs_sim_shifts.png` - Scatter plot

---

### Phase 4: Mechanistic Analysis (Week 5-6)

#### Objectives
- Understand WHY opinions shift
- Identify persuasive argument features
- Test social influence mechanisms

#### Analyses

**4.1 Behavioral Fingerprints (Clustering)**
```python
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Represent each person as 50-dimensional shift vector
shift_profiles = treatment[post_cols].values - treatment[pre_cols].values

# Reduce dimensions
pca = PCA(n_components=5)
shift_pca = pca.fit_transform(shift_profiles)

# Cluster into 5 "deliberator types"
kmeans = KMeans(n_clusters=5, random_state=42)
treatment['deliberator_type'] = kmeans.fit_predict(shift_pca)

# Profile each type
for i in range(5):
    cluster = treatment[treatment['deliberator_type'] == i]
    print(f"\nType {i} (n={len(cluster)}):")
    print(f"  Party: {cluster['PARTYID3'].value_counts(normalize=True).to_dict()}")
    print(f"  Avg shift: {cluster['shift_magnitude'].mean():.1f}")
    print(f"  Shift direction: {cluster['shift_net'].mean():.2f}")
```

**4.2 Group Composition Effects**
```python
# Compute group diversity
group_stats = treatment.groupby('GROUP').agg({
    'PARTYID3': lambda x: len(x.unique()),  # Party diversity
    'AGE': 'std',  # Age spread
    'shift_magnitude': 'mean'  # Group avg shift
}).rename(columns={'PARTYID3': 'party_diversity', 'AGE': 'age_diversity'})

# Test: Do diverse groups shift more?
from scipy.stats import spearmanr
r, p = spearmanr(group_stats['party_diversity'], group_stats['shift_magnitude'])
print(f"Party diversity → Shift magnitude: r={r:.3f}, p={p:.3f}")
```

**4.3 Argument Persuasiveness Analysis**
```python
# Use LLM to rate arguments
def score_argument(argument, recipient_party, question):
    """Rate how persuasive argument is to recipient"""
    prompt = f"""You are a {recipient_party} voter. Rate this argument on a 0-10 scale:

Argument: "{argument}"
Question: {question}

How persuasive is this to you? (0=not at all, 10=very persuasive)

Score:"""
    
    response = call_llm('gpt-4o-mini', prompt, temperature=0.2)
    return int(re.search(r'\d+', response).group())

# Test cross-party persuasiveness
rep_args = [args from Republican agents]
dem_args = [args from Democrat agents]

# Rep args → Dem recipients
cross_party_scores = []
for arg in rep_args:
    score = score_argument(arg, 'Democrat', question)
    cross_party_scores.append(score)

print(f"Cross-party persuasiveness: {np.mean(cross_party_scores):.2f}")
```

**Deliverables**:
- `results/deliberator_types.csv` - Cluster profiles
- `results/group_composition_effects.csv` - Diversity correlations
- `results/argument_persuasiveness.csv` - Cross-party ratings
- `figures/behavioral_fingerprints.png` - PCA visualization

---

## Publication Plan

### Paper Structure

**Title Options**:
1. "Simulating Deliberative Democracy with LLMs: Evidence from America in One Room"
2. "Can AI Predict Opinion Shifts in Face-to-Face Deliberation?"
3. "Demographic Fingerprints of Opinion Change: An LLM Simulation Study"

**Abstract** (200 words):
```
We present the first large-scale validation of large language models (LLMs) as 
simulators of face-to-face deliberation, using America in One Room (n=3,842), 
a national deliberative poll on policy issues. We address three questions: 
(1) Can LLMs predict individual opinion shifts given demographics and initial opinions? 
(2) Which demographic factors correlate with shift patterns? 
(3) Can LLM agents reproduce empirical deliberation dynamics?

Our supervised models achieve R²=0.XX in predicting post-deliberation opinions, 
with party affiliation and initial extremism as strongest predictors. We identify 
five distinct "deliberator types" via clustering, with [Type X] showing [pattern]. 
Agent-based simulations using GPT-4 reproduce partisan depolarization observed in 
real data (r=0.XX correlation between real and simulated shifts).

However, LLM-generated arguments show [limitation/bias], suggesting [implication]. 
Our findings demonstrate that LLMs can serve as scalable proxies for studying 
deliberation under certain conditions, with applications to democratic innovation, 
polarization research, and human-AI collaboration. We release our simulation 
framework to enable future research.
```

**Key Contributions**:
1. First validation of LLM deliberation simulation on large-scale real data
2. Identification of demographic predictors of opinion shift
3. Novel "behavioral fingerprinting" via clustering
4. Open-source simulation framework

### Target Venues

**Tier 1 (Primary Targets)**:
1. **NeurIPS** - Main conference or Datasets & Benchmarks track
   - Fit: Novel application, validation methodology
   - Emphasis: Prediction accuracy, simulation metrics

2. **FAccT** (ACM Conference on Fairness, Accountability, Transparency)
   - Fit: Democratic deliberation, political bias in LLMs
   - Emphasis: Societal impact, fairness considerations

3. **ICLR**
   - Fit: LLM agents, social simulation
   - Emphasis: Mechanistic understanding

**Tier 2 (Backup)**:
4. **CHI** (Computer-Human Interaction)
   - Fit: Human-AI collaboration in decision-making
   
5. **CSCW** (Computer-Supported Cooperative Work)
   - Fit: Group deliberation, computer-mediated discussion

6. **IC²S²** (Computational Social Science)
   - Fit: Perfect for interdisciplinary work

**Journals**:
7. **PNAS** (if findings particularly striking)
8. **EPJ Data Science**
9. **Computational Communication Research**

---

## Resource Requirements

### Compute
- **LLM API costs**: 
  - Full simulation (1,367 agents × 50 questions): ~$500-1,500 with GPT-4
  - Or ~$50-150 with GPT-4o-mini / GPT-3.5
  - Or FREE with open models (Llama 3, Mistral) on local GPU

### Timeline
- Phase 1 (EDA): 1-2 weeks
- Phase 2 (Prediction): 1-2 weeks
- Phase 3 (Simulation): 2-3 weeks (LLM calls take time)
- Phase 4 (Analysis): 1-2 weeks
- Writing: 2-3 weeks
- **Total**: 8-10 weeks (~2.5 months)

### Team
- Minimum: 1 person (you)
- Ideal: 2 people (1 code, 1 analysis/writing)

---

## Key Limitations & How to Address

### Limitation 1: Not Traditional Delphi
**Issue**: This is face-to-face deliberation, not anonymous statistical feedback

**How to address in paper**:
- Frame as "deliberative polling" not "Delphi"
- Emphasize contribution: Demographics + large sample
- Acknowledge: Different mechanism than Delphi (discussion vs stats)
- Position: Broader study of "consensus formation processes"

### Limitation 2: No Discussion Transcripts
**Issue**: Don't know what participants actually said

**How to address**:
- LLMs generate plausible arguments (not actual ones)
- Validate on outcomes (shifts) not process (arguments)
- Acknowledge: Simulation of "process-like" not "actual process"
- Future work: Collect discussion data to improve simulation

### Limitation 3: Single Event (Not Iterative)
**Issue**: Only 2 time points (pre/post weekend), not multiple rounds

**How to address**:
- Focus on "opinion change" not "iterative convergence"
- Compare to control group (shows effect of deliberation)
- Acknowledge: Cannot study round-by-round dynamics
- Position: First step toward modeling iterative processes

---

## Success Criteria

### Minimum Viable Results (Publishable)

1. ✅ **Prediction**: R² ≥ 0.4 for supervised models
2. ✅ **Simulation**: r ≥ 0.3 correlation (real vs sim shifts)
3. ✅ **Demographics**: Identify ≥3 significant predictors
4. ✅ **Depolarization**: LLMs reproduce direction of party convergence

### Stretch Goals

1. 🎯 **High accuracy**: R² ≥ 0.6
2. 🎯 **Strong simulation**: r ≥ 0.5
3. 🎯 **Causal mechanism**: Validate social influence via group composition
4. 🎯 **Novel discovery**: Find "behavioral fingerprints" not in original paper
5. 🎯 **Multi-LLM**: Compare GPT-4, Claude, Llama performance

---

## Immediate Next Steps (This Week)

### Day 1 (Today)
- [ ] Read Fishkin et al. (2021) APSR paper fully
- [ ] Set up conda environment
- [ ] Install dependencies (pandas, sklearn, openai, etc.)

### Day 2-3
- [ ] Load dataset, verify structure
- [ ] Compute basic shift statistics
- [ ] Replicate paper's Figure 2 (depolarization)

### Day 4-5
- [ ] Build baseline prediction model (Ridge regression)
- [ ] Test single LLM agent on 1 question (prototype)
- [ ] Estimate full simulation costs

### Day 6-7
- [ ] Create Phase 1 analysis notebook
- [ ] Draft methods section
- [ ] Plan Phase 2 experiments

---

## Decision: Accept This Dataset?

**Recommendation**: ✅ **YES, proceed with America in One Room**

**Why**:
1. ✅ Large sample (3,842) enables robust analysis
2. ✅ Rich demographics (party, age, race, gender) - your core need
3. ✅ Published in top venue (APSR) with full methodology
4. ✅ Control group enables causal inference
5. ✅ 50 matched question pairs provide statistical power
6. ⚠️ Not pure Delphi, but still studies consensus formation

**Trade-off accepted**: 
- You cannot study Delphi-specific mechanisms (statistical feedback)
- BUT you CAN study demographic predictors of opinion change
- This was your original research goal: "behavioral fingerprints"

**Frame the research as**:
> "LLM Simulation of Discussion-Based Consensus Formation"
> NOT "LLM Simulation of the Delphi Method"

**Bottom line**: This dataset is excellent for your research goals. The fact it's not traditional Delphi is a minor limitation, not a deal-breaker. Start Phase 1 now!

