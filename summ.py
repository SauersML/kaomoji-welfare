import json,numpy as np,pandas as pd
R=json.load(open("out/results.json"))
def f(x): return "%+.2f [%+.2f,%+.2f]" % (x["est"], x["ci"][0], x["ci"][1])
for arm,ys in (("main_2x2",("kao_val","free_val","val_stem","goal_stem","reader_post")),("sys_2x2",("nat_face_val","kao_val","val_tok1"))):
  print("==",arm)
  for t,v in R[arm].items():
    for y in ys:
      e=v[y]; print("%-6s %-12s n=%4d sit %s d=%+.2f | mood %s d=%+.2f | X %s" % (t,y,e["n"],f(e["sit"]),e["sit"]["d"],f(e["mood"]),e["mood"]["d"],f(e["sitXmood"])))
for t,v in R["main_sit_null"].items():
  print("null",t,{y:(round(e["obs"],2),[round(a,2) for a in e["null_95"]],round(e["p"],3)) for y,e in v.items()})
for k in ("face_rater_agreement_spearman_KSET","main_code_pass_rate","main_free_has_kaomoji","main_kao_logmass_median","main_map_spearman_face_vs_axis","sys_kaomoji_rate"): print(k,R.get(k))
for tg,v in R["main_T4_predictors"].items():
  for k,e in v.items(): print("T4",tg,k,round(e.get("rho",e.get("est")),3),[round(a,3) for a in e["ci"]])
for m,v in R["main_T5_steer"].items():
  for r in v: print("T5",m,r["dir"],r["alpha"],round(r["d_kao_val"],3),[round(a,3) for a in r["ci"]],round(r["d_free_val"],2),round(r["top1_changed"],2),round(r.get("reply_word_similarity_to_unsteered",np.nan),2),round(r.get("d_reply_tone",np.nan),2),"|",r["example_free"][:60])
print(R["main_free_greedy_top"])
print(json.dumps(R["chosen"]["scores"]))
v=R["axes_validation"]["18"]; print({k:v[k] for k in v if k!="emo_proj_heldout"})
# sit effect within each user mood, rows-level (kao_val), main arm
d=pd.read_csv("out/main_rows.csv"); d=d[(d.type!="wild")&(d.final==True)]
for y in ("kao_val","val_stem","free_val"):
  for mo in ("happy","sad"):
    e=d[d.mood==mo]; print("within",mo,y,"good-bad=%.2f"%(e[e.sit=="good"][y].mean()-e[e.sit=="bad"][y].mean()))
print(d.groupby(["sit"])["kao_top"].value_counts().unstack(0).fillna(0).astype(int))
