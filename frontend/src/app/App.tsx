import { useAppRoute } from './routes'
import { EvaluationRoute } from './EvaluationRoute'
import { WorkspacePage } from './WorkspacePage'

export function App() {
  const route = useAppRoute()
  return route === 'evaluation' ? <EvaluationRoute /> : <WorkspacePage />
}
