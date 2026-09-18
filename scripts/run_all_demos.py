import sys
sys.path.insert(0, ".")
from anvesha.agent import get_controller

c = get_controller()
demos = [
    (["samples/demo_single_multispectral.tif"],
     "Describe the land-cover and major objects visible in this image."),
    (["samples/demo_single_optical.png"],
     "Highlight the water body referred to in the query."),
    (["samples/demo_change_2020.tif", "samples/demo_change_2024.tif"],
     "What changed between these two dates, and where did the change occur?"),
    (["samples/demo_change_2020.tif", "samples/demo_change_2024.tif"],
     "Has the built-up area increased, decreased, or remained unchanged?"),
    (["samples/demo_pair_optical.tif", "samples/demo_pair_sar.tif"],
     "Use the optical and SAR images together to identify built-up and water-covered regions."),
    (["samples/isro_cartosat2s_optical.tif", "samples/isro_risat_sar.tif"],
     "Use the optical and SAR images together to identify built-up and water-covered regions."),
]
for imgs, q in demos:
    r = c.run(imgs, q)
    print(f"[{r.selected_task}] conf={r.confidence:.2f}")
    print("  Q:", q)
    print("  A:", r.answer[:230].replace("\n", " "))
