"""River Digital Twin: heavy-metal transport via suspended sediment (Rhine at Lobith, Maas at Eijsden).

Data flow:

    readings.csv --> replay.Replayer --> messaging.Broker --> twin.DigitalTwin --> twin.StateStore --> dashboard
                                                                 |
                                         softsensor / quality / monitoring / sampling
"""

__version__ = "0.1.0"
