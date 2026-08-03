import { AnalysisView } from '@/components/app/analysis-view';

// Static route: wins over the [[...uid]] catch-all both in `next dev` and in
// the static export (exported as analysis.html; the FastAPI SPA fallback maps
// /analysis onto it).
export default function Page() {
  return <AnalysisView />;
}
