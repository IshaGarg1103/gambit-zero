# Restored charger-required policy

At the user’s request, external power is required by default again.
The January restart explicitly enables the startup check and disconnect watchdog.
A fully charged battery with external power connected passes the check.

The temporary-output cap waiver is retained, as the user asked to undo only
the charger changes. The 20 GiB free-disk reserve, memory safeguards,
8-hour invocation limit and cumulative transfer allowance remain active.
The launcher is one-shot (KeepAlive=false), stopping after January.

Validation: 36 power, temporary-storage and archive tests passed.
No Git commands were run.
