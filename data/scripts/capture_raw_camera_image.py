from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from ids_peak import ids_peak
from ids_peak import ids_peak_ipl_extension
from ids_peak_ipl import ids_peak_ipl


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
SAVE_DIR = PROJECT_ROOT / "raw_camera"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture one image from the first detected IDS camera.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SAVE_DIR,
        help="Directory where the captured image is written (default: raw_camera).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = (PROJECT_ROOT / output_dir).resolve()

    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / f"capture_{datetime.now():%Y%m%d_%H%M%S}.png"

    ids_peak.Library.Initialize()

    try:
        dm = ids_peak.DeviceManager.Instance()
        dm.Update()
        if dm.Devices().empty():
            raise RuntimeError("No IDS camera detected.")

        device = dm.Devices()[0].OpenDevice(ids_peak.DeviceAccessType_Control)
        nodemap = device.RemoteDevice().NodeMaps()[0]

        if device.DataStreams().empty():
            raise RuntimeError("No data stream available.")
        stream = device.DataStreams()[0].OpenDataStream()

        try:
            nodemap.FindNode("AcquisitionMode").SetCurrentEntry("SingleFrame")
        except Exception:
            pass

        try:
            nodemap.FindNode("TriggerSelector").SetCurrentEntry("ExposureStart")
            nodemap.FindNode("TriggerMode").SetCurrentEntry("Off")
        except Exception:
            pass

        payload_size = nodemap.FindNode("PayloadSize").Value()
        num_buffers = stream.NumBuffersAnnouncedMinRequired()
        for _ in range(num_buffers):
            buf = stream.AllocAndAnnounceBuffer(payload_size)
            stream.QueueBuffer(buf)

        try:
            nodemap.FindNode("TLParamsLocked").SetValue(1)
        except Exception:
            pass

        stream.StartAcquisition()
        nodemap.FindNode("AcquisitionStart").Execute()
        nodemap.FindNode("AcquisitionStart").WaitUntilDone()

        buffer = stream.WaitForFinishedBuffer(5000)

        image = ids_peak_ipl_extension.BufferToImage(buffer)
        try:
            image_out = image.ConvertTo(
                ids_peak_ipl.PixelFormatName_BGRa8,
                ids_peak_ipl.ConversionMode_Fast,
            )
        except Exception:
            image_out = image.ConvertTo(
                ids_peak_ipl.PixelFormatName_Mono8,
                ids_peak_ipl.ConversionMode_Fast,
            )

        stream.QueueBuffer(buffer)

        ids_peak_ipl.ImageWriter.Write(str(file_path), image_out)
        print(f"Capture succeeded. Saved image to: {file_path.resolve()}")

        try:
            nodemap.FindNode("AcquisitionStop").Execute()
            nodemap.FindNode("AcquisitionStop").WaitUntilDone()
        except Exception:
            pass

        stream.StopAcquisition(ids_peak.AcquisitionStopMode_Default)
        stream.Flush(ids_peak.DataStreamFlushMode_DiscardAll)

        for announced_buffer in stream.AnnouncedBuffers():
            stream.RevokeBuffer(announced_buffer)

        try:
            nodemap.FindNode("TLParamsLocked").SetValue(0)
        except Exception:
            pass

    finally:
        ids_peak.Library.Close()


if __name__ == "__main__":
    main()
