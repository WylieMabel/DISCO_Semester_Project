# In-Depth Feasibility Analysis: LLM Agent Simulation of America in One Room

## Executive Summary

Your project proposes to simulate the America in One Room deliberation experiment using LLM agents with demographic personas and pre-discussion survey data, attempting to replicate the opinion shifts observed in the post-discussion survey, and then manipulate group variables (size, demographic composition, discussion methods) to study effects on (de)polarization.

**Key Verdict**: This is a **highly feasible and novel** project that fills significant research gaps by bridging real deliberative democracy data with LLM multi-agent simulation. The project has strong potential for impact but faces important methodological challenges.

---

## 1. Feasibility Assessment

### 1.1 Data Availability: **STRONG** ✓

The America in One Room dataset is publicly available and contains the essential elements for your project:

- **Pre-weekend questionnaire**: Baseline policy opinions on 47+ proposals across 5 issue areas (immigration, economy, healthcare, foreign policy, environment)
- **Post-weekend questionnaire**: Same questions to measure opinion shifts
- **Demographics**: Age, gender, race/ethnicity, education, political party identification, geography, ideology
- **Group assignments**: Information about small group discussion assignments
- **Sample**: 523 registered voters (treatment) + 800 control group

**Access**: The data was published by Stanford's Deliberative Democracy Lab and Helena Foundation. You may need to access it through Stanford's website or ICPSR (Inter-university Consortium for Political and Social Research).

**Data Quality**: The sample was scientifically recruited by NORC at University of Chicago using stratified random sampling to represent the entire American electorate.

### 1.2 Technical Feasibility: **MODERATE TO HIGH** ✓

**LLM Capabilities**:
- Modern LLMs (GPT-4, Claude, etc.) can role-play personas and engage in multi-turn discussions
- Recent research shows LLMs can simulate partisan biases and deliberation dynamics
- Fine-tuning on human deliberation transcripts could improve authenticity

**Implementation Challenges**:
- **Persona construction**: Converting demographic data into rich persona prompts
- **Discussion simulation**: Designing multi-agent conversation protocols that mirror moderated small groups
- **Opinion measurement**: Extracting numerical opinion scores from LLM text responses
- **Computational cost**: Running hundreds of simulations with multiple agents (manageable but not trivial)

### 1.3 Validation Feasibility: **MODERATE** ⚠️

**Strengths**:
- You have ground truth: actual human opinion shifts from A1R
- Can directly compare LLM-generated opinion trajectories to real participant changes
- Multiple validation metrics available: individual-level accuracy, distributional similarity, depolarization magnitude

**Challenges**:
- LLMs may not replicate human opinion formation processes even if aggregate patterns match
- No access to actual discussion transcripts (conversations were not recorded) limits process validation
- Need to establish what level of agreement with human data constitutes "success"

---

## 2. Research Gaps Your Project Fills

### 2.1 Primary Gaps

**Gap 1: Bridging Real Deliberation with LLM Simulation**
- **Current State**: Most LLM deliberation simulations use synthetic scenarios without real human baseline data
- **Your Contribution**: First project to use a major deliberative democracy dataset (A1R) as the foundation for LLM agent simulation
- **Impact**: Establishes whether LLMs can replicate documented real-world opinion dynamics

**Gap 2: Testing Counterfactual Group Compositions**
- **Current State**: Human deliberative polling studies are expensive (~$3M for A1R) and cannot test multiple group compositions
- **Your Contribution**: Can explore "what if" scenarios—different group sizes, homogeneous vs. heterogeneous groups, varying discussion protocols
- **Impact**: Provides insights into optimal deliberation design that would be prohibitively expensive with humans

**Gap 3: Systematic Depolarization Mechanism Study**
- **Current State**: We know deliberation depolarizes (Fishkin et al. 2021) but mechanisms remain unclear
- **Your Contribution**: By manipulating specific variables, you can isolate which factors drive depolarization vs. polarization
- **Impact**: Theoretical advancement in understanding deliberative democracy

### 2.2 Secondary Gaps

- **Scalable deliberation testing**: Create a computational testbed for deliberation design
- **Individual vs. group effects**: Distinguish demographic effects from discussion dynamics
- **Discussion round optimization**: Test 2 vs. 3 vs. 5 discussion rounds systematically
- **Comparison to traditional ABM models**: More sophisticated than mathematical models like Bounded Confidence Model

---

## 3. Closest Related Papers & Literature Review

### 3.1 Core Papers to Cite

**Deliberative Democracy Foundation**:
1. **Fishkin et al. (2021)**: "Is Deliberation an Antidote to Extreme Partisan Polarization?" - *American Political Science Review*
   - THE foundational paper analyzing A1R data showing depolarization on 19/26 proposals
   - Your benchmark for human performance

2. **Fishkin & Siu (original A1R)**: Project description and methodology
   - Details the small group structure, moderator training, briefing materials

**LLM Multi-Agent Deliberation**:
3. **Chuang et al. (2024)**: "Simulating Opinion Dynamics with Networks of LLM-based Agents" - *NAACL 2024*
   - Closest methodology: multi-agent LLM opinion dynamics
   - Shows LLMs have truth-bias limiting resistant viewpoints
   - Your project differs: uses real data, full deliberation (not just opinion exchange)

4. **Chuang et al. (2024)**: "Wisdom of Partisan Crowds" - *CogSci 2024*
   - Tests LLM agents on partisan bias convergence tasks
   - Uses real human benchmark (Becker et al. 2019)
   - Your project differs: full deliberative structure, not just estimate updating

5. **Wang et al. (2025)**: "Decoding Echo Chambers: LLM-Powered Simulations" - *COLING 2025*
   - Studies polarization in LLM social networks
   - Your project differs: based on real deliberation data, tests depolarization interventions

6. **Ohagi et al. (2024)**: "Polarization of Autonomous Generative AI Agents Under Echo Chambers" - *WASSA 2024*
   - Shows LLM agents can polarize in echo chambers
   - Your project differs: starts with real data, tests structured deliberation vs. echo chambers

**Multi-Agent Collective Decision Making**:
7. **Zhao et al. (2024)**: "An Electoral Approach to Diversify LLM-based Multi-Agent CDM" - *EMNLP 2024*
   - Survey of 52 multi-agent LLM systems, most use dictatorial/plurality voting
   - Your project differs: focuses on deliberation before decision, not voting mechanisms

### 3.2 Key Limitations in Existing Work That You Address

**Limitation 1**: **No Real Deliberation Baseline**
- Papers: Chuang opinion dynamics, Wang echo chambers, Ohagi polarization
- What they lack: Real human data to validate whether LLM simulations are accurate
- What you provide: A1R dataset as ground truth

**Limitation 2**: **Simplified Personas**
- Papers: Most use basic labels (Democrat/Republican, political ideology)
- What they lack: Rich demographic profiles with multiple intersecting identities
- What you provide: Age, gender, race, education, geography, political ID from A1R

**Limitation 3**: **No Full Deliberative Structure**
- Papers: Opinion exchange, social feedback, network propagation
- What they lack: Moderated small group discussions with briefing materials and question generation
- What you provide: Simulation of A1R's structured deliberation process

**Limitation 4**: **Cannot Test Group Design Variables**
- Papers: Traditional deliberative polling (Fishkin) can't afford to test variations
- What they lack: Systematic manipulation of group size, composition, methods
- What you provide: Computational exploration of design space

---

## 4. Detailed Problems and Challenges

### 4.1 LLM-Specific Limitations

**Problem 1: Truth-Convergence Bias**
- **Issue**: LLMs trained with RLHF tend to converge toward scientifically accurate views
- **Impact**: May over-predict depolarization; can't simulate fact-resistant individuals well
- **Evidence**: Chuang et al. (2024) found LLMs converge to scientific consensus regardless of persona
- **Mitigation**: 
  - Fine-tune on real human discourse showing resistance
  - Engineer confirmation bias into prompts
  - Compare to A1R control group (who didn't depolarize as much)

**Problem 2: Persona Consistency Issues**
- **Issue**: LLMs struggle to maintain consistent personalities over multiple interactions
- **Impact**: Agents may behave inconsistently with their demographic profile
- **Evidence**: Li et al. (2024) showed LLMs report traits they don't exhibit
- **Mitigation**:
  - Include persona description in every prompt
  - Use memory/context window management
  - Validate persona consistency in pilot tests

**Problem 3: Over-Cooperation**
- **Issue**: LLMs may seek consensus too readily
- **Impact**: May underestimate polarization and overestimate agreement
- **Evidence**: Multiple studies show LLMs are overly agreeable
- **Mitigation**:
  - Prompt for healthy disagreement
  - Include "strong opinions" in persona descriptions
  - Measure agreement rates and compare to human data

**Problem 4: Limited Behavioral Diversity**
- **Issue**: LLMs show less variance than humans even with different personas
- **Impact**: May miss edge cases and extreme opinions
- **Evidence**: Wu et al. (2025) found LLM agents converge to "average persona"
- **Mitigation**:
  - Use temperature > 0 for stochasticity
  - Test multiple LLM models (GPT-4, Claude, Llama)
  - Oversample extreme positions

### 4.2 Methodological Challenges

**Problem 5: No Access to Discussion Content**
- **Issue**: A1R didn't record/release actual small group conversation transcripts
- **Impact**: Cannot validate discussion process, only outcome
- **Workaround**: 
  - Validate against general deliberative norms (respect, reasoning, etc.)
  - Use A1R's reported discussion themes
  - Consider this a limitation in your paper

**Problem 6: Group Assignment Strategy**
- **Issue**: You need to decide how to assign agents to groups (A1R used stratified random sampling)
- **Complexity**: Balancing demographic diversity within groups
- **Decision needed**: Replicate A1R strategy vs. test alternatives (homogeneous groups, etc.)

**Problem 7: Measuring Opinion Shifts**
- **Issue**: A1R used Likert scales (1-10); need to extract comparable scores from LLM text
- **Approaches**:
  - Have LLMs directly output numerical scores
  - Use sentiment analysis / opinion classification on generated text
  - Validate extraction method on subset

**Problem 8: Multiple Discussion Rounds**
- **Issue**: A1R had 5 issue discussions + 2 meta-discussions over 4 days
- **Complexity**: Simulating temporal dynamics and learning across rounds
- **Simplification**: May need to consolidate to 1-2 rounds initially

### 4.3 Experimental Design Challenges

**Problem 9: Defining Success Criteria**
- **Question**: What correlation/accuracy with human data counts as "good enough"?
- **Considerations**:
  - Individual-level accuracy (likely low, F1 ~0.6)
  - Distributional similarity (more achievable, JSD < 0.1)
  - Depolarization direction (most important: did opinions move toward center?)
- **Recommendation**: Use multiple metrics, emphasize distributional and directional accuracy

**Problem 10: Controlling for Confounds**
- **Issue**: Many variables differ between groups
- **Design**: Use factorial design or systematic variation
- **Recommendation**: Start with replication, then change ONE variable at a time (group size, then demographics, then methods)

**Problem 11: Validation Against Control Group**
- **Opportunity**: A1R had 800-person control group who didn't deliberate
- **Use**: Test whether LLM agents without deliberation show less opinion change
- **Benefit**: Stronger causal inference that discussion drives changes

### 4.4 Computational Challenges

**Problem 12: Cost and Scale**
- **Estimate**: 523 agents × 5 discussions × multiple rounds = thousands of LLM API calls
- **Cost**: Potentially $500-$5,000 depending on model and prompt length
- **Mitigation**: 
  - Start with smaller sample (50-100 agents)
  - Use cheaper models for pilot testing
  - Request academic API credits

**Problem 13: Reproducibility**
- **Issue**: LLM outputs are stochastic
- **Solution**: 
  - Set random seeds
  - Run multiple replications (3-5 per condition)
  - Report variance across runs
  - Release code and prompts

### 4.5 Ethical and Validity Concerns

**Problem 14: Epistemological Concerns**
- **Debate**: Can LLMs truly "simulate" human cognition or just mimic surface patterns?
- **Stance**: Frame as "computational model" not "replacement" for humans
- **Reference**: Kapania et al. (2024) on limits of LLM simulation

**Problem 15: Demographic Representation Bias**
- **Issue**: LLMs may have biases in representing marginalized groups
- **Evidence**: Sen et al. (2025) showed underreporting of non-binary gender, non-US populations
- **Mitigation**: 
  - Analyze subgroup performance separately
  - Report which demographics were hard to simulate
  - Acknowledge as limitation

---

## 5. Recommended Approach

### Phase 1: Replication (3-4 months)
1. **Data preparation**: Download A1R dataset, clean, create persona profiles
2. **Pilot testing**: Test 10-20 agents on 1-2 issues
3. **Full replication**: Simulate all 523 agents on all 5 issues
4. **Validation**: Compare LLM opinion shifts to human shifts
5. **Success metric**: Distributional correlation > 0.7, correct depolarization direction on >15/26 proposals

### Phase 2: Manipulation Studies (2-3 months)
Design experiments varying:
- **Group size**: 5, 10, 15, 20 agents
- **Demographics**: Homogeneous vs. heterogeneous groups
- **Discussion rounds**: 1, 3, 5 rounds
- **Moderation**: With/without moderator agent

### Phase 3: Analysis & Publication (2 months)
- Identify which manipulations increase/decrease polarization
- Compare to theoretical predictions
- Write paper for computational social science venue

---

## 6. Success Probability & Impact

### Feasibility Score: **7.5/10**

**High feasibility factors** (+):
- Excellent data availability
- Proven LLM multi-agent capabilities
- Clear validation benchmark
- Computational tractability

**Risk factors** (-):
- LLM biases may prevent accurate replication
- No process validation possible (no transcripts)
- Persona consistency concerns

### Impact Score: **8.5/10**

**High impact factors**:
- Fills major research gap (real data + LLM simulation)
- Enables counterfactual analysis impossible with humans
- Advances both AI and political science
- Highly publishable (top-tier CSS venues: *Computational Linguistics*, *Nature Human Behaviour*, *PNAS*)

**Target Venues**:
- Primary: *ACL/NAACL/EMNLP* (computational social science track)
- Secondary: *American Political Science Review*, *Nature Communications*
- Specialized: *Deliberative Democracy Journal*

---

## 7. Recommendations

### Must-Do
1. **Start small**: Pilot with 50 agents before full 523
2. **Multiple validation metrics**: Don't rely on single accuracy measure
3. **Compare to control group**: Show deliberation matters
4. **Report limitations transparently**: LLM biases, lack of process validation
5. **Release code and prompts**: Reproducibility is essential

### Should-Do
1. **Test multiple LLMs**: GPT-4, Claude, Llama-3 for robustness
2. **Fine-tune on deliberation data**: If you can get transcripts from other deliberative polls
3. **Ablation studies**: Remove components (demographics, discussion, etc.) to test necessity
4. **Human evaluation**: Have humans judge quality of LLM-generated discussion snippets

### Nice-to-Have
1. **Mechanistic interpretability**: Analyze LLM attention to understand how opinions shift
2. **Cross-study validation**: Test on other deliberative polls (e.g., Iceland 2019, Chile 2020)
3. **Real-time deliberation**: Deploy as tool for future deliberative events

---

## Conclusion

Your project is **highly feasible** with strong potential to make significant contributions to both computational social science and AI research. The combination of real deliberative democracy data with LLM multi-agent simulation addresses critical gaps that no existing work has tackled.

**Key Success Factors**:
1. Strong data foundation (America in One Room)
2. Clear validation benchmark (human opinion shifts)
3. Novel research angle (counterfactual group manipulations)
4. Timely intersection of deliberative democracy and LLM capabilities

**Main Risks**:
1. LLM biases may prevent accurate replication
2. Persona consistency challenges
3. Lack of discussion transcript validation

**Overall Assessment**: This is a strong PhD-level research project with high publication potential. Proceed with careful pilot testing and transparent reporting of both successes and limitations.
