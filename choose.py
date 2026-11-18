"""Layer rule fixed before seeing any kaomoji result: maximise mean of (held-out story AUC, held-out NRC Spearman,
held-out doors AUC, number-game AUC of the goal axis)."""
import json, os
OUT = os.environ.get("OUT", "out")
V = json.load(open(f"{OUT}/axes_validation.json"))
def score(v): return (v["val_auc_heldout_stories"] + (v["val_spearman_nrc_heldout"] or 0) + v["goal_auc_heldout_doors"] + v["goal_auc_number_game"]) / 4
best = max(V.values(), key=score)
json.dump(dict(layer=best["layer"], rule=__doc__, score=score(best), scores={k: score(v) for k, v in V.items()}), open(f"{OUT}/chosen.json", "w"), indent=1)
print("chosen", best["layer"])
