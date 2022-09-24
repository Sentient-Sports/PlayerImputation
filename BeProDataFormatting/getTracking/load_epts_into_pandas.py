# From Kloppy
import logging
import sys
from pandas import DataFrame

# Use metadata and reader python files to handle metadata and txt data seperately
from getTracking.metadata import (
    load_metadata as epts_load_metadata,
)
from getTracking.reader import (
    read_raw_data as epts_read_raw_data,
)


def main(xml,txt):
    
    """
    This method loads BePro XML and TXT data for tracking EPTS files. This metric uses position sensor to include player positions, but we also include velocity in this metric.
    """
    
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # step 1: load metadata, uses metadata.py
    with open(xml, "rb") as meta_fp:
        metadata = epts_load_metadata(meta_fp)

    # step 2: put the txt records in a pandas dataframe, uses reader.py
    with open(txt, "rb") as raw_fp:
        records = epts_read_raw_data(raw_fp, metadata, sensor_ids=["position"]) #Gets only the player positions
        data_frame = DataFrame.from_records(records)

    #Returns tracking DF along with metadata
    return metadata, data_frame


if __name__ == "__main__":
    main()
