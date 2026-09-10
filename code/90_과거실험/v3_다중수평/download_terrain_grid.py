"""Fetch a 3x3 public DEM elevation grid and derive local terrain slope/aspect."""
from __future__ import annotations
import json, math, urllib.parse, urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
LAT,LON=35.1601,126.8515
delta=0.001
points=[(LAT+dy,LON+dx) for dy in (-delta,0,delta) for dx in (-delta,0,delta)]
params={"latitude":",".join(str(p[0]) for p in points),"longitude":",".join(str(p[1]) for p in points)}
url="https://api.open-meteo.com/v1/elevation?"+urllib.parse.urlencode(params,safe=",")
with urllib.request.urlopen(url,timeout=60) as r: values=json.load(r)["elevation"]
z=[values[i:i+3] for i in range(0,9,3)]
dx_m=111320*math.cos(math.radians(LAT))*delta; dy_m=110540*delta
dzdx=(z[1][2]-z[1][0])/(2*dx_m); dzdy=(z[2][1]-z[0][1])/(2*dy_m)
slope=math.degrees(math.atan(math.sqrt(dzdx**2+dzdy**2)))
aspect=(math.degrees(math.atan2(dzdx,-dzdy))+360)%360
out={"center":{"latitude":LAT,"longitude":LON},"spacing_degrees":delta,"grid_elevation_m":z,
     "terrain_slope_deg_dem_approx":slope,"terrain_aspect_deg_dem_approx":aspect,
     "limitation":"Approximate surrounding ground terrain from DEM; not PV module tilt/azimuth or rooftop surface.",
     "source_url":url}
(ROOT/"external"/"gwangju_terrain_grid.json").write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps(out,ensure_ascii=False,indent=2))
