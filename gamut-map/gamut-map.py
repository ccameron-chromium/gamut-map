import numpy as np

def OKLCH_to_OKLab(LCH):
  L = LCH[0]
  c = LCH[1]
  h = LCH[2]

  L = L # L is still L
  a = c * np.cos(h * np.pi / 180)
  b = c * np.sin(h * np.pi / 180)
  return np.array([L, a, b])

def OKLab_to_OKLCH(Lab):
  L = Lab[0]
  a = Lab[1]
  b = Lab[2]

  L = L
  C = np.sqrt(a ** 2 + b ** 2)
  H = np.arctan2(b, a) * 180 / np.pi
  if H < 0:
    H = H + 360
  return np.array([L, C, H])

def OKLab_to_XYZD65(OKLab):
  LMStoXYZ = np.array([
      [  1.2268798733741557,  -0.5578149965554813,   0.28139105017721583 ],
      [ -0.04057576262431372,  1.1122868293970594,  -0.07171106666151701 ],
      [ -0.07637294974672142, -0.4214933239627914,   1.5869240244272418  ]]);
  OKLabtoLMS = np.array([
      [ 0.99999999845051981432,  0.39633779217376785678,   0.21580375806075880339  ],
      [ 1.0000000088817607767,  -0.1055613423236563494,   -0.063854174771705903402 ],
      [ 1.0000000546724109177,  -0.089484182094965759684, -1.2914855378640917399   ]]);
  LMSnl = np.dot(OKLabtoLMS, np.array(OKLab))
  return np.dot(LMStoXYZ, np.power(LMSnl, 3));

def XYZD65_to_OKLab(OKLab):
  XYZtoLMS = np.array([
      [ 0.8190224432164319,    0.3619062562801221,   -0.12887378261216414  ],
      [ 0.0329836671980271,    0.9292868468965546,     0.03614466816999844 ],
      [ 0.048177199566046255,  0.26423952494422764,    0.6335478258136937  ]]);
  LMStoOKLab = np.array([
      [  0.2104542553,   0.7936177850,  -0.0040720468 ],
      [  1.9779984951,  -2.4285922050,   0.4505937099 ],
      [  0.0259040371,   0.7827717662,  -0.8086757660 ]]);
  LMS = np.dot(XYZtoLMS, np.array(OKLab));
  return np.dot(LMStoOKLab, np.power(LMS, 1/3))

def LinearP3_to_XYZD65(rgb):
  M = np.array([
      [0.4865709486482162, 0.26566769316909306, 0.1982172852343625],
      [0.2289745640697488, 0.6917385218365064,  0.079286914093745],
      [0.0000000000000000, 0.04511338185890264, 1.043944368900976]])
  xyz = np.dot(M, np.array(rgb))
  return xyz

def XYZD65_to_LinearP3(xyz):
  M = np.array([
      [ 2.493496911941425,   -0.9313836179191239, -0.40271078445071684],
      [-0.8294889695615747,   1.7626640603183463,  0.023624685841943577],
      [ 0.03584583024378447, -0.07617238926804182, 0.9568845240076872]])
  rgb = np.dot(M, np.array(xyz))
  return rgb

def LinearSRGB_to_XYZD65(rgb):
  M = np.array([
      [ 0.41239079926595934, 0.357584339383878,   0.1804807884018343  ],
      [ 0.21263900587151027, 0.715168678767756,   0.07219231536073371 ],
      [ 0.01933081871559182, 0.11919477979462598, 0.9505321522496607  ]])
  xyz = np.dot(M, np.array(rgb))
  return xyz

def XYZD65_to_LinearSRGB(xyz):
  M = np.array([
      [  3.2409699419045226,  -1.537383177570094,   -0.4986107602930034  ],
      [ -0.9692436362808796,   1.8759675015077202,   0.04155505740717559 ],
      [  0.05563007969699366, -0.20397695888897652,  1.0569715142428786  ]])
  rgb = np.dot(M, np.array(xyz))
  return rgb

def SRGB_to_Linear_1D(x):
  abs_x = np.abs(x)
  if abs_x < 0.04045:
    return x / 12.92;
  else:
    return np.sign(x) * (np.power((abs_x + 0.055) / 1.055, 2.4))

def Linear_to_Encoded_1D(x):
  abs_x = np.abs(x);
  if abs_x > 0.0031308:
  	return np.sign(x) * (1.055 * np.power(abs_x, 1/2.4) - 0.055)
  else:
    return 12.92 * x

def Encoded_to_Linear(rgb):
  return np.array([
      SRGB_to_Linear_1D(rgb[0]),
      SRGB_to_Linear_1D(rgb[1]),
      SRGB_to_Linear_1D(rgb[2])])

def Linear_to_Encoded(rgb):
  return np.array([
      Linear_to_Encoded_1D(rgb[0]),
      Linear_to_Encoded_1D(rgb[1]),
      Linear_to_Encoded_1D(rgb[2])])

def SRGB_to_XYZD65(rgb):
  return LinearSRGB_to_XYZD65(Encoded_to_Linear(rgb))

def XYZD65_to_SRGB(xyz):
  srgb_linear = Linear_to_Encoded(XYZD65_to_LinearSRGB(xyz))

def IsInUnitInterval(rgb):
  return rgb[0] >= 0 and rgb[0] <= 1 and \
         rgb[1] >= 0 and rgb[1] <= 1 and \
         rgb[2] >= 0 and rgb[2] <= 1

def GamutMapInOKLCH(rgb, RGB_to_XYZD65_fn, XYZD65_to_RGB_fn):
  OKLCH = OKLab_to_OKLCH(XYZD65_to_OKLab((RGB_to_XYZD65_fn(rgb))))
  L = OKLCH[0]
  H = OKLCH[2]
  C_min = 0
  C_max = OKLCH[1]
  # Do 100 bisections iterations, rather than trying to terminate early.
  i = 0
  while True:
    C_midpoint = (C_max+C_min)/2
    OKLCH_midpoint = np.array([L, C_midpoint, H])
    rgb_midpoint = XYZD65_to_RGB_fn(OKLab_to_XYZD65(OKLCH_to_OKLab(OKLCH_midpoint)))
    if i == 100:
      return rgb_midpoint
    i += 1
    if (IsInUnitInterval(rgb_midpoint)):
      C_min = C_midpoint
    else:
      C_max = C_midpoint
  return rgb_midpoint

def GamutMapInOKLab(rgb, RGB_to_XYZD65_fn, XYZD65_to_RGB_fn):
  OKLab = XYZD65_to_OKLab((RGB_to_XYZD65_fn(rgb)))
  L = OKLab[0]
  a = OKLab[1]
  b = OKLab[2]
  # Scale the length of (a, b), because inverse trigonometric functions are
  # always a sign that life has taken a wrong turn.
  r_min = 0.0
  r_max = 1.0
  # Do 100 bisections iterations, rather than trying to terminate early.
  i = 0
  while True:
    r_midpoint = (r_max+r_min)/2
    OKLab_midpoint = np.array([L, r_midpoint*a, r_midpoint*b])
    rgb_midpoint = XYZD65_to_RGB_fn(OKLab_to_XYZD65(OKLab_midpoint))
    if i == 100:
      return rgb_midpoint
    i += 1
    if (IsInUnitInterval(rgb_midpoint)):
      r_min = r_midpoint
    else:
      r_max = r_midpoint
  return rgb_midpoint

GamutMap = GamutMapInOKLab

rgb_p3 = np.array([1.0, 0.0, 0.0])
print(rgb_p3)
rgb_srgb = XYZD65_to_LinearSRGB(LinearP3_to_XYZD65(rgb_p3))
print(rgb_srgb)
rgb_srgb_mapped = GamutMap(rgb_srgb, LinearSRGB_to_XYZD65, XYZD65_to_LinearSRGB)
print(rgb_srgb_mapped)
rgb_srgb_mapped = Linear_to_Encoded(rgb_srgb_mapped)
print('rgb(%f%% %f%% %f%%)' % (100*rgb_srgb_mapped[0], 100*rgb_srgb_mapped[1], 100*rgb_srgb_mapped[2]))
