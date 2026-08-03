import { APP_CONFIG_DEFAULTS } from '@/app-config';

interface LayoutProps {
  children: React.ReactNode;
}

export default function Layout({ children }: LayoutProps) {
  const { companyName, logo, logoDark } = APP_CONFIG_DEFAULTS;

  return (
    <>
      <header className="fixed top-0 left-0 z-50 hidden w-full flex-row items-center justify-between p-6 md:flex">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img src={logo} alt={`${companyName} Logo`} className="block h-6 w-auto dark:hidden" />
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={logoDark ?? logo}
          alt={`${companyName} Logo`}
          className="hidden h-6 w-auto dark:block"
        />
        <span className="text-muted-foreground font-mono text-xs font-bold tracking-wider uppercase">
          AI Interview Platform
        </span>
      </header>

      {children}
    </>
  );
}
