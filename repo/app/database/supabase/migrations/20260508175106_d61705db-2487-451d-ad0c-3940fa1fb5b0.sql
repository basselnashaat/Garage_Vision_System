
CREATE TABLE public.cameras (
  id UUID NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,
  location_name TEXT NOT NULL,
  location_type TEXT NOT NULL,
  installed_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE public.visits (
  id UUID NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,
  camera_id UUID NOT NULL REFERENCES public.cameras(id) ON DELETE CASCADE,
  plate_hash TEXT NOT NULL,
  plate_digits TEXT NOT NULL,
  plate_letters TEXT NOT NULL,
  visited_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX idx_visits_camera ON public.visits(camera_id);
CREATE INDEX idx_visits_visited_at ON public.visits(visited_at DESC);
CREATE INDEX idx_visits_plate_hash ON public.visits(plate_hash);

CREATE TABLE public.plate_scores (
  id UUID NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,
  visit_id UUID NOT NULL REFERENCES public.visits(id) ON DELETE CASCADE,
  digit_length_score NUMERIC NOT NULL DEFAULT 0,
  repeat_score NUMERIC NOT NULL DEFAULT 0,
  sequential_score NUMERIC NOT NULL DEFAULT 0,
  low_number_score NUMERIC NOT NULL DEFAULT 0,
  total_score NUMERIC NOT NULL DEFAULT 0
);
CREATE INDEX idx_plate_scores_visit ON public.plate_scores(visit_id);

CREATE TABLE public.segment_scores (
  id UUID NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,
  visit_id UUID NOT NULL REFERENCES public.visits(id) ON DELETE CASCADE,
  segment TEXT NOT NULL,
  confidence NUMERIC NOT NULL DEFAULT 0,
  segment_score NUMERIC NOT NULL DEFAULT 0
);
CREATE INDEX idx_segment_scores_visit ON public.segment_scores(visit_id);

CREATE TABLE public.purchasing_power (
  id UUID NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,
  visit_id UUID NOT NULL REFERENCES public.visits(id) ON DELETE CASCADE,
  plate_weight NUMERIC NOT NULL DEFAULT 0,
  segment_weight NUMERIC NOT NULL DEFAULT 0,
  final_score NUMERIC NOT NULL DEFAULT 0
);
CREATE INDEX idx_pp_visit ON public.purchasing_power(visit_id);
CREATE INDEX idx_pp_final_score ON public.purchasing_power(final_score DESC);

ALTER TABLE public.cameras ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.visits ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.plate_scores ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.segment_scores ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.purchasing_power ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Public read cameras" ON public.cameras FOR SELECT USING (true);
CREATE POLICY "Public read visits" ON public.visits FOR SELECT USING (true);
CREATE POLICY "Public read plate_scores" ON public.plate_scores FOR SELECT USING (true);
CREATE POLICY "Public read segment_scores" ON public.segment_scores FOR SELECT USING (true);
CREATE POLICY "Public read purchasing_power" ON public.purchasing_power FOR SELECT USING (true);
