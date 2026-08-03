-- Post-interview analysis produced by the LLM grader
-- (local_voice_ai/interview_analysis.py) after each call ends.
--
-- rubric_scores holds one entry per rubric:
--   [{"key": "technical_knowledge", "name": "Technical Knowledge",
--     "weight": 0.30, "score": 4, "justification": "..."}, ...]
-- overall_score is the weighted average of the 1-5 rubric scores scaled
-- to 0-100.

CREATE TABLE IF NOT EXISTS public.interview_analyses
(
    id serial NOT NULL,
    candidate_uid uuid NOT NULL,
    transcript_id integer,
    rubric_scores jsonb NOT NULL,
    overall_score numeric(5,2) NOT NULL,
    recommendation text NOT NULL,
    summary text,
    strengths jsonb,
    areas_of_concern jsonb,
    model text,
    created_at timestamp with time zone DEFAULT now(),
    CONSTRAINT interview_analyses_pkey PRIMARY KEY (id),
    CONSTRAINT interview_analyses_candidate_uid_fkey FOREIGN KEY (candidate_uid)
        REFERENCES public.candidates (uid) MATCH SIMPLE
        ON UPDATE NO ACTION
        ON DELETE NO ACTION,
    CONSTRAINT interview_analyses_transcript_id_fkey FOREIGN KEY (transcript_id)
        REFERENCES public.interview_transcripts (id) MATCH SIMPLE
        ON UPDATE NO ACTION
        ON DELETE NO ACTION,
    CONSTRAINT interview_analyses_overall_score_check
        CHECK (overall_score >= 0 AND overall_score <= 100),
    CONSTRAINT interview_analyses_recommendation_check
        CHECK (recommendation IN ('strong_yes', 'yes', 'maybe', 'no'))
);

CREATE INDEX IF NOT EXISTS interview_analyses_candidate_uid_idx
    ON public.interview_analyses (candidate_uid);
