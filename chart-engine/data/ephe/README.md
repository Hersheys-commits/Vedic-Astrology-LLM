# Swiss Ephemeris data files

The engine runs without any files here, falling back to the **Moshier**
analytical ephemeris that is compiled into pyswisseph. For modern dates its
positions agree with the full Swiss Ephemeris to well under an arc-second,
which is far below any astrologically meaningful difference.

For maximum precision (or dates far outside 1800-2400), download the `.se1`
files from the Astrodienst FTP mirror and drop them in this directory:

    sepl_18.se1     planets, 1800-2400
    semo_18.se1     moon,    1800-2400
    seas_18.se1     asteroids (optional)

The engine detects them automatically on startup and reports
`"ephemeris_mode": "swiss"` in `/health` and in every chart's `provenance`
block. Swiss Ephemeris is dual-licensed (AGPL / commercial): check the licence
before shipping the data files in a product.
