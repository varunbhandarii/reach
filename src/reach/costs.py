"""Configured usage estimates; not an invoice or a billing balance."""


def estimate(cfg,perception=None,voice=None,readback=None):
    prices = cfg.get('costs',{})
    perception,voice,readback = perception or {},voice or {},readback or {}
    images = int(perception.get('images_sent',0))
    seconds = float(voice.get('streamed_audio_s',0))
    sam = images*float(prices.get('sam_per_image',.0025))
    speech = seconds*float(prices.get('voice_per_hour',.18))/3600
    spark = float(readback.get('cost_usd',0))
    return {'estimated_usd':sam+speech+spark,'sam_usd':sam,'voice_usd':speech,'spark_usd':spark,
            'sam_images_sent':images,'voice_audio_seconds':seconds,
            'spark_input_tokens':int(readback.get('input_tokens',0)),
            'spark_output_tokens':int(readback.get('output_tokens',0)),
            'note':'Configured standard-price estimate; credits, failed requests and provider billing may differ.'}
