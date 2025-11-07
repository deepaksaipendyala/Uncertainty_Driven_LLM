# Documentation Index

Welcome to the Uncertainty-Guided Hierarchical Program Repair project documentation.

## Getting Started (Read First)

### 1. [Quick Start Guide](quick_start_guide.md)
**For: First-time users**

Quick setup instructions, directory structure, and how to begin implementation.

Key sections:
- Architecture overview diagram
- Setup steps for Llama models
- Key directories explanation
- Daily workflow recommendations

**Start here if**: You want to jump right in.

---

### 2. [Implementation Plan](implementation_plan.md) ⭐ MAIN CHECKLIST
**For: Active developers**

Comprehensive task list with checkboxes for tracking progress. This is your primary working document.

Structure:
- 7 Phases, 18 major sections
- Each task has detailed implementation notes
- Code skeletons and expected outputs
- Success criteria and deliverables

**Use this for**: Daily task tracking and implementation details.

---

### 3. [Project Summary](project_summary.md)
**For: High-level understanding**

Executive overview of the project, novel contributions, and expected impact.

Key sections:
- 5 novel research contributions
- System architecture diagram
- Implementation structure
- Publication targets
- Technical innovations

**Read this for**: Understanding the big picture and research contributions.

---

## Technical Details

### 4. [Architecture Overview](architecture.md)
**For: Understanding system design**

Component layers, data flow, and extensibility points.

Key sections:
- LLM providers
- Uncertainty estimators
- Repair strategies
- Messaging layer
- Pipeline adapters
- Evaluation suite

**Read this for**: Deep dive into system design.

---

### 5. [Evaluation Plan](evaluation_plan.md)
**For: Metrics and benchmarks**

Details on benchmarks, metrics, and evaluation procedures.

Key sections:
- 4 benchmark descriptions (GSM8K, TruthfulQA, Defects4J, HumanEval)
- Novel metrics definitions
- Evaluation procedures
- Expected results

**Read this for**: Understanding how the system is evaluated.

---

## Project Management

### 6. [ROADMAP.md](../ROADMAP.md)
**For: High-level progress tracking**

Phase-by-phase roadmap with status indicators and timelines.

Key sections:
- Phase structure and status
- Novel contributions checklist
- Success metrics tracking
- Resource setup checklist
- Publication timeline

**Use this for**: Tracking overall project progress.

---

## Reading Sequence

### For New Contributors:
1. **Quick Start Guide** - Setup and orientation
2. **Project Summary** - Understand the goals
3. **Implementation Plan** - Start implementing
4. **Architecture** - Deep dive as needed

### For Reviewers/Advisors:
1. **Project Summary** - High-level overview
2. **Architecture** - System design
3. **Evaluation Plan** - Validation strategy
4. **Implementation Plan** - Implementation details

### For Users:
1. **Quick Start Guide** - Setup
2. **Architecture** - How it works
3. **Evaluation Plan** - Performance characteristics

---

## Document Relationships

```
                    ┌─────────────────┐
                    │   ROADMAP.md    │
                    │  (High-level)   │
                    └────────┬────────┘
                             │
                   ┌─────────┴─────────┐
                   │                   │
         ┌─────────▼────────┐  ┌──────▼──────────┐
         │  Project Summary │  │  Quick Start    │
         │  (What & Why)    │  │  (How to Begin) │
         └─────────┬────────┘  └──────┬──────────┘
                   │                   │
         ┌─────────▼────────────────────▼───────────┐
         │      Implementation Plan (MAIN)          │
         │      (Detailed Task Checklist)           │
         └─────────┬───────────────────┬────────────┘
                   │                   │
         ┌─────────▼────────┐  ┌──────▼──────────┐
         │   Architecture   │  │  Evaluation     │
         │   (Design)       │  │  (Metrics)      │
         └──────────────────┘  └─────────────────┘
```

---

## Key Files Summary

| File | Purpose | When to Use |
|------|---------|-------------|
| `quick_start_guide.md` | Setup & first steps | Starting the project |
| `implementation_plan.md` | Main checklist with tasks | Daily development |
| `project_summary.md` | Executive overview | Understanding goals |
| `architecture.md` | System design | Deep technical understanding |
| `evaluation_plan.md` | Benchmarks & metrics | Planning experiments |
| `ROADMAP.md` | Progress tracking | Weekly status updates |

---

## Quick Navigation

### I want to...

**...start implementing**
→ Go to [Implementation Plan](implementation_plan.md), Section 1.1

**...understand the architecture**
→ Read [Architecture Overview](architecture.md)

**...see the big picture**
→ Read [Project Summary](project_summary.md)

**...set up my environment**
→ Follow [Quick Start Guide](quick_start_guide.md)

**...know what to evaluate**
→ Check [Evaluation Plan](evaluation_plan.md)

**...track progress**
→ Update [ROADMAP.md](../ROADMAP.md)

**...understand a specific component**
→ Search [Implementation Plan](implementation_plan.md) for relevant section

---

## Document Maintenance

### Updating Progress
1. Check off tasks in `implementation_plan.md` as you complete them
2. Update phase status in `ROADMAP.md` periodically
3. Add notes/insights to `ROADMAP.md` as you discover them

### Adding New Sections
1. Add detailed tasks to `implementation_plan.md`
2. Update `ROADMAP.md` if it's a new phase
3. Reference from `quick_start_guide.md` if user-facing

### Before Submission
1. Ensure all checkboxes in `implementation_plan.md` are complete
2. Update `project_summary.md` with final results
3. Generate final result tables for `evaluation_plan.md`

---

## Additional Resources

### External Documentation
- **LogTokU Paper**: https://arxiv.org/abs/2502.00290
- **RepairAgent Paper**: https://arxiv.org/abs/2403.17134
- **LogTokU Repo**: `/home/dpendya/Documents/dlba/logtoku/`
- **RepairAgent Repo**: `/home/dpendya/Documents/dlba/RepairAgent/`

### Project Structure
```
/home/dpendya/Documents/dlba/
├── docs/               ← You are here
├── src/                ← Implementation code
├── tests/              ← Unit tests
├── logtoku/            ← LogTokU source
└── RepairAgent/        ← RepairAgent source
```

---

**Last Updated**: Initial creation
**Version**: 1.0
**Status**: Ready to begin implementation

---

## Contact

For questions or clarifications about the documentation structure, refer to the specific document or the implementation plan for detailed technical guidance.

**Happy Building!** This is a novel, state-of-the-art research project. Follow the implementation plan systematically, check off tasks as you complete them, and track your progress regularly.

